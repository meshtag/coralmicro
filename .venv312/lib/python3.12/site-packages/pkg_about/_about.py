# Copyright (c) 2020 Adam Karpierz
# SPDX-License-Identifier: Zlib

__all__ = ('about', 'about_from_setup')

import sys
from typing import Any
from pathlib import Path
from collections import namedtuple
from functools import partial
from email.utils import getaddresses, parseaddr
from importlib.metadata import metadata as get_metadata
from importlib.metadata import version as get_version
from packaging.version import parse as parse_version
import setuptools  # noqa: F401
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]

if sys.version_info >= (3, 12, 6):
    getaddresses = partial(getaddresses, strict=False)
    parseaddr    = partial(parseaddr,    strict=False)
else: pass  # pragma: no cover

version_info = namedtuple("version_info",
                          ["major", "minor", "micro", "releaselevel", "serial"],
                          module=__package__)


class adict(dict[str, Any]):

    __module__ = __package__

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__getitem__(name)
        except KeyError as exc:
            raise AttributeError(*exc.args) from None

    def __setattr__(self, name: str, value: Any) -> None:
        try:
            self.__setitem__(name, value)
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(*exc.args) from None

    def __delattr__(self, name: str) -> None:
        try:
            self.__delitem__(name)
        except KeyError as exc:
            raise AttributeError(*exc.args) from None

    copy = __copy__ = lambda self: self.__class__(self)


def about(package: str | None = None) -> adict:
    pkg_globals = sys._getframe(1).f_globals
    pkg_globals.pop("__builtins__", None)
    pkg_globals.pop("__cached__",   None)
    if package is None: package = pkg_globals["__package__"]
    metadata = get_metadata(package)
    version = parse_version(get_version(package))
    project_urls = {item.partition(",")[0].strip():
                    item.partition(",")[2].lstrip()
                    for item in metadata.get_all("Project-URL") or []}
    release_levels = __release_levels
    metadata_get = metadata.get  # type: ignore[attr-defined]

    pkg_metadata = adict(
        __title__        = metadata["Name"],
        __version__      = str(version),
        __version_info__ = version_info(
                               major=version.major,
                               minor=version.minor,
                               micro=version.micro,
                               releaselevel=release_levels[
                                   version.pre[0] if version.pre else
                                   "dev"   if version.dev   else
                                   "post"  if version.post  else
                                   "local" if version.local else
                                   "final"],
                               serial=(version.pre[1] if version.pre else
                                       version.dev or version.post
                                       or version.local or 0)),
        __summary__      = metadata_get("Summary"),
        __uri__          = (metadata_get("Home-page")
                            or project_urls.get("Home-page")
                            or project_urls.get("Homepage")
                            or project_urls.get("Home")),
        __author__       = metadata_get("Author"),
        __email__        = None,
        __author_email__ = metadata_get("Author-email"),
        __maintainer__       = metadata_get("Maintainer"),
        __maintainer_email__ = metadata_get("Maintainer-email"),
        __license__      = (metadata_get("License-Expression")
                            or metadata_get("License")),
        __copyright__    = None,
    )
    email = pkg_metadata["__author_email__"] or ""
    names = ", ".join(name for name, _ in getaddresses([email]) if name)
    if names:
        if not pkg_metadata["__author__"]:
            pkg_metadata["__author__"] = names
        else:  # pragma: no cover
            pkg_metadata["__author__"] += ", " + names
    email = pkg_metadata["__maintainer_email__"] or ""
    names = ", ".join(name for name, _ in getaddresses([email]) if name)
    if names:
        if not pkg_metadata["__maintainer__"]:
            pkg_metadata["__maintainer__"] = names
        else:  # pragma: no cover
            pkg_metadata["__maintainer__"] += ", " + names
    pkg_metadata["__email__"] = pkg_metadata["__author_email__"]
    pkg_metadata["__copyright__"] = pkg_metadata["__author__"]

    pkg_globals.update(pkg_metadata)
    pkg_globals.setdefault("__all__", [])
    pkg_globals["__all__"] += list(pkg_metadata.keys())
    return pkg_metadata


def about_from_setup(package_path: Path | str | None = None) -> adict:
    try:
        from setuptools.config.setupcfg import (read_configuration as
                                                read_setupcfg)
    except ImportError:  # pragma: no cover
        from setuptools.config import read_configuration as read_setupcfg
    try:
        from setuptools.config.pyprojecttoml import (read_configuration as
                                                     read_pyproject_toml)
    except ImportError:  # pragma: no cover
        read_pyproject_toml = None  # type: ignore[assignment]
    pkg_globals = sys._getframe(1).f_globals
    package_path = (Path(pkg_globals["__file__"]).resolve().parents[1]
                    if package_path is None else Path(package_path))
    pyproject_path = package_path/"pyproject.toml"
    setup_cfg_path = package_path/"setup.cfg"
    metadata = {}
    if setup_cfg_path.exists():  # pragma: no branch
        metadata.update(read_setupcfg(setup_cfg_path,
                        ignore_option_errors=True).get("metadata", {}))
    if pyproject_path.exists():  # pragma: no branch
        if read_pyproject_toml is not None:
            metadata.update(read_pyproject_toml(pyproject_path,
                            ignore_option_errors=True).get("project", {}))
        else:  # pragma: no cover
            with pyproject_path.open("rb") as file:
                metadata.update(tomllib.load(file).get("project", {}))
    version = parse_version(metadata["version"])
    get, release_levels = __get, __release_levels

    authors     = get(metadata, "authors") or []
    maintainers = get(metadata, "maintainers") or []

    pkg_metadata = adict(
        __title__        = metadata["name"],
        __version__      = str(version),
        __version_info__ = version_info(
                               major=version.major,
                               minor=version.minor,
                               micro=version.micro,
                               releaselevel=release_levels[
                                   version.pre[0] if version.pre else
                                   "dev"   if version.dev   else
                                   "post"  if version.post  else
                                   "local" if version.local else
                                   "final"],
                               serial=(version.pre[1] if version.pre else
                                       version.dev or version.post
                                       or version.local or 0)),
        __summary__      = get(metadata, "description"),
        __uri__          = (get(metadata, "urls", "Home-page")
                            or get(metadata, "urls", "Homepage")
                            or get(metadata, "urls", "Home")
                            or get(metadata, "url")),
        __author__       = (", ".join(item["name"] for item in authors
                                      if item and "name" in item)
                            or get(metadata, "author")),
        __email__        = None,
        __author_email__ = (", ".join((f"{item['name']} <{item['email']}>"
                                       if "name" in item and not parseaddr(
                                          item["email"])[0]
                                       else item["email"])
                                      for item in authors
                                      if item and "email" in item)
                            or get(metadata, "author_email")),
        __maintainer__   = (", ".join(item["name"] for item in maintainers
                                      if item and "name"  in item)
                            or get(metadata, "maintainer")),
        __maintainer_email__ = (", ".join((f"{item['name']} <{item['email']}>"
                                           if "name" in item and not parseaddr(
                                              item["email"])[0]
                                           else item["email"])
                                          for item in maintainers
                                          if item and "email" in item)
                                or get(metadata, "maintainer_email")),
        __license__      = (get(metadata, "license", "text")
                            or get(metadata, "license")),
        __copyright__    = None,
    )
    pkg_metadata["__email__"] = pkg_metadata["__author_email__"]
    pkg_metadata["__copyright__"] = pkg_metadata["__author__"]

    pkg_globals["about"] = pkg_metadata
    pkg_globals.setdefault("__all__", [])
    pkg_globals["__all__"].append("about")
    return pkg_metadata


def __get(metadata: Any, *keys: Any) -> Any:
    for key in keys:
        if isinstance(metadata, dict):
            if key not in metadata:
                return None
        elif isinstance(metadata, (list, tuple)):
            if key >= len(metadata):  # pragma: no cover
                return None
        else:  # pragma: no cover
            return None
        metadata = metadata[key]
    return metadata


__release_levels = dict(
    a     = "alpha",
    b     = "beta",
    rc    = "candidate",
    dev   = "dev",
    post  = "post",
    local = "local",
    final = "final",
)
