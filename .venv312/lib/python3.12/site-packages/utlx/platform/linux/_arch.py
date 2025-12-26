# Copyright (c) 1994 Adam Karpierz
# SPDX-License-Identifier: Zlib

__all__ = ('arch',)


def get_python_arch() -> str | None:
    import sys
    import platform
    machine = platform.machine().lower()
    is_32bits = (sys.maxsize <= 2**32)
    little_endian = sys.byteorder.lower() == "little"

    X86_64      = ("x86_64", "amd64")
    X86_32      = ("i386", "i486", "i586", "i686", "x86", "ia32")
    ARM_64      = ("aarch64", "arm64")
    ARM_64_BE   = ("aarch64_be", "arm64_be")
    ARM_32_SOFT = ("armv6l", "armv5tel", "armv4tl", "armv4l", "arml", "armel")
    ARM_32_HARD = ("armv8l", "armv7l")
    PPC_64_LE   = ("ppc64le", "powerpc64le")
    PPC_64      = ("ppc64", "powerpc64")
    PPC_32_LE   = ("ppcle", "powerpcle")
    PPC_32      = ("ppc", "powerpc")
    MIPS_64_LE  = ("mips64el", "mips64le")
    MIPS_64_BE  = ("mips64", "mips64eb")
    MIPS_32_LE  = ("mipsel", "mipsle")
    MIPS_32_BE  = ("mips", "mipseb")
    RISC_V_64   = ("riscv64",)
    S390X       = ("s390x",)

    # x86
    if machine in X86_64:
        arch = "x86_64" if not is_32bits else "x86"
    elif machine in X86_32:
        arch = "x86"
    # ARM 64-bit
    elif machine in ARM_64 and not is_32bits:
        suffix = "" if little_endian else "be"
        arch = "aarch64" + suffix
    elif machine in ARM_64_BE and not is_32bits:
        suffix = "be"
        arch = "aarch64" + suffix
    # ARM 32-bit
    elif machine in ARM_32_SOFT:
        suffix = "le"
        arch = "arm" + suffix
    elif (machine in ARM_32_HARD or (machine in ARM_64 and is_32bits)):
        # hf = hard-float,little-endian, le = soft-float,little-endian
        suffix = "hf" if has_fpu() else "le"
        arch = "arm" + suffix
    elif ((machine.startswith("arm") and machine.endswith("b"))
          or (machine in ARM_64_BE and is_32bits)):
        arch = None  # 32-bit big endian is not supported
    # POWER / ppc
    elif machine in PPC_64_LE or machine in PPC_64:
        suffix = "le" if little_endian else ""
        arch = ("ppc64" if not is_32bits else "ppc") + suffix
    elif machine in PPC_32_LE or machine in PPC_32:
        suffix = "le" if little_endian else ""
        arch = "ppc" + suffix
    # MIPS
    elif machine in MIPS_64_LE or machine in MIPS_64_BE:
        suffix = "le" if little_endian else ""
        arch = ("mips64" if not is_32bits else "mips") + suffix
    elif machine in MIPS_32_LE or machine in MIPS_32_BE:
        suffix = "le" if little_endian else ""
        arch = "mips" + suffix
    # RISC-V
    elif machine in RISC_V_64:
        arch = "riscv64"
    # IBM Z
    elif machine in S390X:
        arch = "s390x"
    else:
        arch = None  # unknown or unsupported

    return arch


def has_fpu() -> bool:
    """
    Detects presence of FPU on ARM:

    1. Try /proc/cpuinfo
    2. Fallback: readelf on Python interpreter
    """
    import sys
    import re
    import subprocess
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                line = line.lower()
                if "vfp" in line or "neon" in line:
                    return True
    except FileNotFoundError:
        pass
    try:
        out = subprocess.check_output(["readelf", "-A", sys.executable],
                                      stderr=subprocess.DEVNULL, text=True)
        if re.search(r"vfp|neon", out, re.IGNORECASE):
            return True
    except (OSError, subprocess.CalledProcessError):
        pass
    return False


arch = get_python_arch()
