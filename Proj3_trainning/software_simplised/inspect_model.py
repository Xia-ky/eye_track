#!/usr/bin/env python3
"""无需 GPU 即可检查 JSON 所描述的 SEE-D 网络。"""

from __future__ import annotations

import argparse
from pathlib import Path

from esda.config import ConfigError, load_config
from esda.models.plan import (
    build_layer_plan,
    count_trainable_parameters,
    format_layer_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
    except ConfigError as error:
        print(f"配置错误：{error}")
        return 2
    parameter_count = count_trainable_parameters(config)
    print(format_layer_plan(build_layer_plan(config), parameter_count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
