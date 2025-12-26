# Copyright (c) 2016 Adam Karpierz
# SPDX-License-Identifier: Zlib

from typing  import Any
from pathlib import Path
import configparser

__all__ = ('make_config',)


def make_config(cfg_fname: str, cfg_section: str | None = None) -> None:
    import sys
    from pathlib import Path
    from functools import partial
    fglobals = sys._getframe(1).f_globals
    fglobals.pop("__builtins__", None)
    fglobals.pop("__cached__",   None)
    if cfg_section is None: cfg_section = fglobals["__package__"]
    cfg_path = Path(fglobals["__file__"]).parent/cfg_fname
    fglobals["__all__"] = ("config", "set_config")
    fglobals["config"] = get_config(cfg_path, cfg_section)
    fglobals["set_config"] = partial(set_config, fglobals)


def get_config(cfg_path: Path | str,
               cfg_section: str | None = None) -> configparser.SectionProxy:
    from pathlib import Path
    from configparser import ConfigParser, ExtendedInterpolation, DEFAULTSECT
    cfg_path = Path(cfg_path)
    interpolation = ExtendedInterpolation()
    inline_comment_prefixes = ("#", ";")
    if cfg_section is not None:
        cfg = ConfigParser(interpolation=interpolation,
                           inline_comment_prefixes=inline_comment_prefixes,
                           default_section=cfg_section)
    else:
        cfg = ConfigParser(interpolation=interpolation,
                           inline_comment_prefixes=inline_comment_prefixes)
    cfg.read(cfg_path, encoding="utf-8")
    return cfg[cfg_section if cfg_section is not None else DEFAULTSECT]


def set_config(fglobals: dict[str, Any], **cfg_dict: Any) -> None:
    import sys
    import importlib
    # Update config
    to_update = {key: str(val) for key, val in cfg_dict.items()
                 if val is not None}
    to_remove = {key for key, val in cfg_dict.items() if val is None}
    package_name = fglobals["__package__"]
    config_name  = package_name + ".__config__"
    config = sys.modules[config_name].config
    config.update(to_update)
    for key in to_remove: config.pop(key, None)
    # Reload
    for mod_name in tuple(sys.modules):
        if (mod_name.startswith(package_name + ".")
           and mod_name != config_name):
            del sys.modules[mod_name]
    importlib.reload(sys.modules[package_name])


del Any
del Path
del configparser
