"""YAML config + `section.key=value` command-line overrides."""

import ast
from copy import deepcopy

import yaml


def load_config(path, overrides=()):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return apply_overrides(cfg, overrides)


def apply_overrides(cfg, overrides=()):
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must look like section.key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            low = val.lower()
            val = True if low == "true" else False if low == "false" else val
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return cfg


def dump_config(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(deepcopy(cfg), f, sort_keys=False, allow_unicode=True)
