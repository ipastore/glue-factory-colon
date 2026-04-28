from pathlib import Path

import lightglue
from lightglue.lightglue import LightGlue
import gluefactory
import torch

if __name__ == "__main__":
    experiment = "<name-of-your-trained-experiment>"
    weights_name = "my_lightglue_weights.pth"
    out = Path(lightglue.__file__).parent / "weights" / weights_name
    model = gluefactory.load_experiment(experiment)
    torch.save(model.matcher.state_dict(), out)
    matcher = LightGlue(weights=weights_name, input_dim=256)