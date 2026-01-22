#!/usr/bin/env python3
import argparse, subprocess, re, sys, shutil
import numpy as np
from pathlib import Path

def q_to_R(qw,qx,qy,qz):
    xx,yy,zz = qx*qx,qy*qy,qz*qz
    xy,xz,yz = qx*qy,qx*qz,qy*qz
    wx,wy,wz = qw*qx,qw*qy,qw*qz
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
    ], dtype=np.float64)

def parse_cameras(path):
    cams={}
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip() or ln.startswith('#'): continue
            p=ln.split()
            if len(p)<4: continue
            cam_id=int(p[0]); width=int(p[2]); height=int(p[3])
            cams[cam_id]=(width,height)
    return cams

def parse_points3d(path):
    pts={}
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            if not ln.strip() or ln.startswith('#'): continue
            p=ln.split()
            if len(p)<7: continue
            pid=int(p[0]); X=float(p[1]); Y=float(p[2]); Z=float(p[3])
            pts[pid]=(X,Y,Z)
    return pts

def parse_images(path):
    imgs=[]
    with open(path,'r',encoding='utf-8') as f:
        lines=[ln.rstrip('\n') for ln in f if ln.strip() and not ln.startswith('#')]
    i=0
    while i<len(lines):
        p=lines[i].split()
        if len(p)<10: i+=1; continue
        d={
            'image_id':int(p[0]),
            'qw':float(p[1]),'qx':float(p[2]),'qy':float(p[3]),'qz':float(p[4]),
            'tx':float(p[5]),'ty':float(p[6]),'tz':float(p[7]),
            'camera_id':int(p[8]),
            'name':" ".join(p[9:])
        }
        obs=[]
        if i+1<len(lines):
            q=lines[i+1].split()
            for k in range(0,len(q),3):
                try:
                    u=float(q[k]); v=float(q[k+1]); pid=int(float(q[k+2]))
                    obs.append((u,v,pid))
                except: pass
        d['points2d']=obs
        imgs.append(d)
        i+=2
    return imgs

def ffmpeg_extract_scaled_from_video(video_path, frame_index, out_path, W=720, H=540):
    vf = f"select='eq(n\\,{frame_index})',scale={W}:{H}"
    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-y",
           "-i", str(video_path), "-vf", vf, "-vsync","0","-frames:v","1", str(out_path)]
    subprocess.run(cmd, check=True)

def ffmpeg_scale_image(src_path, dst_path, W=720, H=540):
    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-y",
           "-i", str(src_path), "-vf", f"scale={W}:{H}", str(dst_path)]
    subprocess.run(cmd, check=True)

def ffmpeg_preextract_all_frames(video_path, tmp_dir, W=720, H=540):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg","-hide_banner","-loglevel","error","-y",
           "-i", str(video_path),
           "-vf", f"scale={W}:{H}",
           "-vsync","0","-start_number","0",
           str(tmp_dir / "%08d.png")]
    subprocess.run(cmd, check=True)

def draw_overlay(base_path, points_uv, src_size, out_path, radius=2, color=(255,0,0,140)):
    from PIL import Image, ImageDraw
    Wsrc,Hsrc = src_size
    Wdst,Hdst = 720,540
    sx = Wdst/float(Wsrc); sy = Hdst/float(Hsrc)
    base = Image.open(base_path).convert("RGBA")
    overlay = Image.new("RGBA", (Wdst,Hdst), (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    for (u,v) in points_uv:
        x = int(round(u*sx)); y = int(round(v*sy))
        if 0 <= x < Wdst and 0 <= y < Hdst:
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
    Image.alpha_composite(base, overlay).convert("RGB").save(out_path)

def process_one_map(map_id:int, map_dir:Path, out_root:Path, args, pre_frames_dir:Path=None):
    cameras_txt = map_dir / "cameras.txt"
    images_txt  = map_dir / "images.txt"
    points3D_txt= map_dir / "points3D.txt"
    if not (cameras_txt.exists() and images_txt.exists() and points3D_txt.exists()):
        print(f"[{map_id}] Saltado: faltan ficheros en {map_dir}")
        return

    cams = parse_cameras(cameras_txt)
    imgs = parse_images(images_txt)
    pts3d= parse_points3d(points3D_txt)

    if args.keyframes_in_map_dir:
        outdir = map_dir / "keyframes"
    else:
        outdir = out_root / f"{map_id}"
    outdir.mkdir(parents=True, exist_ok=True)

    numeric_name = re.compile(r"^[0-9]+$")

    for img in imgs:
        img_id=img['image_id']; cam_id=img['camera_id']; name=img['name']
        if cam_id not in cams:
            print(f"[{map_id}] WARN: camera_id {cam_id} no está en cameras.txt; salto {img_id}")
            continue
        Wsrc,Hsrc=cams[cam_id]

        # número <n>
        if args.number_from == "image_id":
            n = img_id; src_is_index = bool(numeric_name.match(name))
        elif args.number_from == "name":
            if numeric_name.match(name): n, src_is_index = int(name), True
            else: n, src_is_index = img_id, False
        else:
            if numeric_name.match(name): n, src_is_index = int(name), True
            else: n, src_is_index = img_id, False

        # --- Depth sparse (optional) ---
        pts_uv = []
        need_depth = not args.keyframes_in_map_dir
        if need_depth or args.overlay:
            R = q_to_R(img['qw'],img['qx'],img['qy'],img['qz'])  # world->cam
            t = np.array([img['tx'],img['ty'],img['tz']], dtype=np.float64)
            depth = np.zeros((Hsrc,Wsrc), dtype=np.float32) if need_depth else None
            for (u,v,pid) in img['points2d']:
                if pid == -1 or pid not in pts3d: continue
                Xw,Yw,Zw = pts3d[pid]
                Xc,Yc,Zc = R @ np.array([Xw,Yw,Zw]) + t
                if Zc <= 0: continue
                ui,vi = int(round(u)), int(round(v))
                if 0 <= ui < Wsrc and 0 <= vi < Hsrc:
                    if need_depth:
                        depth[vi,ui] = Zc
                    if args.overlay:
                        pts_uv.append((u,v))
            if need_depth:
                np.save(outdir / f"depth_{n}.npy", depth)

        # --- Keyframe base 720x540 (sin puntos) ---
        base_dst = outdir / f"Keyframe_{n}.png"
        if src_is_index:
            if args.video:
                if args.video_strategy == "batch" and pre_frames_dir is not None:
                    src = pre_frames_dir / f"{int(name):08d}.png"
                    if src.exists():
                        shutil.copy2(src, base_dst)
                    else:
                        print(f"[{map_id}] WARN faltó preframe {src.name}; caigo a perframe")
                        ffmpeg_extract_scaled_from_video(args.video, int(name)+int(args.frame_offset), base_dst, 720, 540)
                else:
                    ffmpeg_extract_scaled_from_video(args.video, int(name)+int(args.frame_offset), base_dst, 720, 540)
            else:
                print(f"[{map_id}] WARN NAME numérico y sin --video; no extraigo Keyframe.")
        else:
            if args.frames_dir:
                src = Path(args.frames_dir) / name
                if src.exists():
                    ffmpeg_scale_image(src, base_dst, 720, 540)
                else:
                    print(f"[{map_id}] WARN no existe {src}; no copio Keyframe.")
            else:
                print(f"[{map_id}] WARN NAME no numérico y sin --frames_dir; no copio Keyframe.")

        # --- Overlay opcional ---
        if args.overlay and base_dst.exists() and pts_uv:
            color = tuple(map(int, args.overlay_color.split(",")))
            draw_overlay(base_dst, pts_uv, (Wsrc,Hsrc), outdir / f"KeyframeOverlay_{n}.png",
                         radius=args.overlay_radius, color=color)

        depth_status = "SKIP" if args.keyframes_in_map_dir else "OK"
        print(f"[{map_id}] n={n} depth={depth_status} img={'OK' if base_dst.exists() else '—'} overlay={'ON' if args.overlay else 'OFF'}")

def main():
    ap=argparse.ArgumentParser(description="Todos los mapas: depth_<n>.npy + Keyframe_<n>.png (720x540); overlay opcional; extracción batch.")
    ap.add_argument("--maps_root", required=True)
    ap.add_argument("--out_root",  required=False, help="Salida por mapa (map_id/). Si usas --keyframes_in_map_dir se ignora.")
    ap.add_argument("--keyframes_in_map_dir", action="store_true",
                    help="Solo exporta Keyframe_*.png dentro de <maps_root>/<map_id>/keyframes/ (sin depth). Útil para extraer solo keyframes.")
    ap.add_argument("--video",     help="Vídeo original (para NAME numérico).")
    ap.add_argument("--frames_dir")
    ap.add_argument("--frame_offset", type=int, default=0)
    ap.add_argument("--number_from", choices=["auto","name","image_id"], default="auto")

    # Overlay opcional
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--overlay_radius", type=int, default=2)
    ap.add_argument("--overlay_color", default="255,0,0,140")

    # NUEVO: estrategia de extracción
    ap.add_argument("--video_strategy", choices=["batch","perframe"], default="batch",
                    help="batch=decodifica una vez todo el vídeo; perframe=una llamada ffmpeg por frame.")
    ap.add_argument("--preextract_dir", help="Dónde guardar frames 720x540 si usas batch (por defecto: <out_root>/_preframes)")

    args=ap.parse_args()

    maps_root = Path(args.maps_root)
    if args.out_root is None and not args.keyframes_in_map_dir:
        ap.error("--out_root es obligatorio salvo que uses --keyframes_in_map_dir")
    out_root = Path(args.out_root) if args.out_root else None
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)
        valid_dirs = [d for d in maps_root.iterdir() if d.is_dir() and d.name.isdigit()]
        valid_dirs.sort(key=lambda d: int(d.name))
        map_dirs = [(int(d.name), d) for d in valid_dirs]
    if not map_dirs:
        print("No se encontraron mapas en", maps_root)
        sys.exit(1)

    # Pre-extracción si procede
    pre_frames_dir = None
    if args.video and args.video_strategy == "batch" and not args.keyframes_in_map_dir:
        pre_frames_dir = Path(args.preextract_dir) if args.preextract_dir else (out_root / "_preframes")
        if not pre_frames_dir.exists() or not any(pre_frames_dir.iterdir()):
            print("[preextract] Decodificando el vídeo una sola vez a 720x540…")
            ffmpeg_preextract_all_frames(args.video, pre_frames_dir, 720, 540)
        else:
            print("[preextract] Reutilizando frames ya preextraídos:", pre_frames_dir)
    elif args.video_strategy == "batch" and args.keyframes_in_map_dir:
        print("[preextract] Modo keyframes: no se preextrae el vídeo completo; se decodifican solo los frames necesarios.")

    for mid, mdir in map_dirs:
        process_one_map(mid, mdir, out_root if out_root else maps_root, args, pre_frames_dir)

    print("\nDONE.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
