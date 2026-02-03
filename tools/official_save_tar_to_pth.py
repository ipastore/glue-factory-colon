#https://github.com/cvg/glue-factory/issues/118

import torch
from lightglue.lightglue import LightGlue

import gluefactory

if __name__ == "__main__":
    experiment = "long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin"
    weights_name = "long26_gt_pos_null_10_lg1e-03_lr5e-05_5bin.pth"
    model = gluefactory.load_experiment(experiment)
    torch.save(model.matcher.state_dict(), weights_name)
    matcher = LightGlue(weights=weights_name, input_dim=128)