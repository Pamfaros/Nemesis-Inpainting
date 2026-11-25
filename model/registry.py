
import torch


def build_model(config):

    config = dict(config)
    architecture = config.pop("architecture")

    if architecture == "baseline":
        from .model import NemesisUNet

    elif architecture == "hybrid":
        from .model_hybrid import NemesisUNet

    else:
        raise ValueError(
            f"Unknown architecture: {architecture}"
        )

    return NemesisUNet(**config)


def load_model(weights, device, legacy_config=None):

    checkpoint = torch.load(
        weights,
        map_location="cpu",
        weights_only=True
    )

    config = checkpoint.get("model_config")

    if config is None:
        if legacy_config is None:
            raise ValueError(
                "Checkpoint has no architecture configuration"
            )

        config = legacy_config

    model = build_model(config)

    model.load_state_dict(
        checkpoint["model"],
        strict=True
    )

    return model.to(device).eval()
