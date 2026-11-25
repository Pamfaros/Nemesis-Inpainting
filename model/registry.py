import torch


def build_model(config):
    config = dict(config)
    architecture = config.pop("architecture")
    if architecture in ("baseline", "manhwa"):
        from .model_baseline import NemesisUNet

        return NemesisUNet(**config, color_contours=architecture == "manhwa")
    if architecture == "hybrid":
        from .model_hybrid import NemesisUNet

        return NemesisUNet(**config)
    raise ValueError(f"Unknown architecture: {architecture}")


def checkpoint_config(checkpoint, legacy_config=None):
    config = checkpoint.get("model_config", legacy_config)
    if config is not None:
        return config

    # The original pilot stored only its baseline state and epoch.
    state = checkpoint.get("model", {})
    entry = state.get("entry.0.weight")
    if "context.0.0.weight" in state and entry is not None and entry.shape[1] == 7:
        return {"architecture": "baseline", "width": entry.shape[0]}
    raise ValueError("Checkpoint has no architecture configuration")


def initialize_model(model, config, checkpoint):
    source = checkpoint_config(checkpoint)
    state = checkpoint["model"]
    if config["architecture"] == "manhwa" and source == {**config, "architecture": "baseline"}:
        state = dict(state)
        entry = state["entry.0.weight"]
        weights = torch.zeros_like(model.entry[0].weight, device=entry.device)
        weights[:, :entry.shape[1]].copy_(entry)
        state["entry.0.weight"] = weights
    elif source != config:
        raise ValueError(f"Cannot initialize {config} with checkpoint for {source}")
    model.load_state_dict(state, strict=True)


def load_model(weights, device, legacy_config=None):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    model = build_model(checkpoint_config(checkpoint, legacy_config))
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()
