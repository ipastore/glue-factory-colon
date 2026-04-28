#https://github.com/cvg/glue-factory/issues/48

import torch

def extract_checkpoint(checkpoint_path, save_model_path, n_layers):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print("Checkpoint Contents:")
    state_dict = checkpoint.get('model', {})

    matcher_dict = {k.split('matcher.', 1)[1]: v for k, v in state_dict.items() if k.startswith('matcher')}


    if matcher_dict:
        for i in range(n_layers):
            patterns = [
                (f"transformers.{i}.self_attn", f"self_attn.{i}"),
                (f"transformers.{i}.cross_attn", f"cross_attn.{i}")
            ]

            for old_key, new_key in patterns:
                matcher_dict = {k.replace(old_key, new_key) if old_key in k else k: v for k, v in matcher_dict.items()}

    print(matcher_dict.keys())

    torch.save(matcher_dict, save_model_path)

if __name__ == "__main__":
    checkpoint_path = "outputs/training/long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin/checkpoint_best.tar"
    save_model_path = "outputs/training/long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin/long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin_bis.pth"
    n_layers = 9  # Just like in official repo
    extract_checkpoint(checkpoint_path, save_model_path, n_layers)