from pathlib import Path

root = Path(__file__).parent.parent  # top-level directory
HDD_path = Path("/media/student/HDD/nacho/glue-factory/")

DATA_PATH = root / "data/"  # datasets and pretrained weights
TRAINING_PATH = root / "outputs/training/"  # training checkpoints
EVAL_PATH = root / "outputs/results/"  # evaluation results
THIRD_PARTY_PATH = root / "third_party/"  # third-party libraries

ALLOW_PICKLE = False  # allow pickle (e.g. in torch.load)
