import json
import os
from functools import partial

import torch

from .. import encoders
from ..transformer import TransformerModel


def load_model_only_inference(path, ckpt_path_weights, ckpt_path_config, device):
    """
    Loads a saved model from the specified position. This function only restores inference capabilities and
    cannot be used for further training.
    """

    # Load the model state
    model_state = torch.load(
        os.path.join(path, ckpt_path_weights),
        map_location="cpu",
        weights_only=True,
    )

    # Load the config
    with open(os.path.join(path, ckpt_path_config), "r") as f:
        config_sample = json.load(f)

    if (
        (
            "nan_prob_no_reason" in config_sample
            and config_sample["nan_prob_no_reason"] > 0.0
        )
        or (
            "nan_prob_a_reason" in config_sample
            and config_sample["nan_prob_a_reason"] > 0.0
        )
        or (
            "nan_prob_unknown_reason" in config_sample
            and config_sample["nan_prob_unknown_reason"] > 0.0
        )
    ):
        encoder = encoders.NanHandlingEncoder
    else:
        encoder = partial(encoders.NanSafeLinear, replace_nan_by_zero=True)

    n_out = config_sample["max_num_classes"]

    device = device if torch.cuda.is_available() else "cpu:0"
    encoder = encoder(config_sample["num_features"], config_sample["emsize"])

    nhid = config_sample["emsize"] * config_sample["nhid_factor"]
    y_encoder_generator = (
        encoders.get_Canonical(config_sample["max_num_classes"])
        if config_sample.get("canonical_y_encoder", False)
        else encoders.NanSafeLinear
    )

    assert config_sample["max_num_classes"] > 2
    loss = torch.nn.CrossEntropyLoss(
        reduction="none", weight=torch.ones(int(config_sample["max_num_classes"]))
    )

    model = TransformerModel(
        encoder,
        n_out,
        config_sample["emsize"],
        config_sample["nhead"],
        nhid,
        config_sample["nlayers"],
        y_encoder=y_encoder_generator(1, config_sample["emsize"]),
        dropout=config_sample["dropout"],
        efficient_eval_masking=config_sample["efficient_eval_masking"],
    )

    # print(f"Using a Transformer with {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.{2}f} M parameters")

    model.criterion = loss
    module_prefix = "module."
    model_state = {k.replace(module_prefix, ""): v for k, v in model_state.items()}
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    return (float("inf"), float("inf"), model), config_sample  # no loss measured
