import argparse
import ast
import importlib
from pathlib import Path

import torch

def parse_scalar(value):
    value = value.strip()
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

def load_simple_yaml(path):
    """Small YAML subset loader for this project's config files."""
    root = {}
    stack = [(-1, root)]
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Invalid config line: {raw_line}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)
    return root

def deep_update(base, updates):
    for key, value in updates.items():
        cur = base
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = parse_scalar(value)
    return base

def parse_args():
    parser = argparse.ArgumentParser(description="Offline RL training entrypoint")
    parser.add_argument("--model", default="dt", help="Model name under models/, e.g. dt")
    parser.add_argument("--config", default=None, help="Config path. Defaults to configs/<model>.yaml")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values, e.g. --set training.max_steps=100",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config_path = Path(args.config) if args.config else Path("configs") / f"{args.model}.yaml"
    config = load_simple_yaml(config_path)

    overrides = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must use KEY=VALUE format: {item}")
        overrides[key] = value
    deep_update(config, overrides)

    model_name = config.get("model", {}).get("name", args.model)
    module = importlib.import_module(f"models.{model_name}")
    if not hasattr(module, "train"):
        raise AttributeError(f"models/{model_name}.py must expose a train(config) function")
    module.train(config)

if __name__ == "__main__":
    main()