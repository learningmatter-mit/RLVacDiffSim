from pathlib import Path


def _validate_output_name(output_name):
    if output_name is None:
        return
    if not output_name or Path(output_name).name != output_name:
        raise ValueError("output_name must be a non-empty filename, not a path")


def trajectory_filename(output_name, episode, n_episodes):
    """Return the configured trajectory name, preserving legacy names by default."""
    if output_name is None:
        return f"XDATCAR{episode}"

    _validate_output_name(output_name)
    if "{episode}" in output_name:
        return output_name.format(episode=episode)
    if n_episodes == 1:
        return output_name

    path = Path(output_name)
    return f"{path.stem}-{episode}{path.suffix}"


def artifact_filename(output_name, default_name):
    """Apply the trajectory's identifying suffix to a related output artifact."""
    if output_name is None:
        return default_name

    _validate_output_name(output_name)
    output_stem = Path(output_name.replace("{episode}", "all")).stem
    if output_stem.startswith("XDATCAR"):
        tag = output_stem[len("XDATCAR"):]
    else:
        tag = f"-{output_stem}"

    default_path = Path(default_name)
    return f"{default_path.stem}{tag}{default_path.suffix}"
