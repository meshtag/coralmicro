# flake8-in-file-ignores: noqa: E305,F401,F821

# Copyright (c) 1994 Adam Karpierz
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import ctypes as ct
from ctypes import CDLL as DLL
try:
    from _ctypes import dlclose  # type: ignore[attr-defined,unused-ignore]
except ImportError:  # pragma: no cover
    dlclose = lambda handle: None
from ctypes import CFUNCTYPE as CFUNC

from ...ctypes import POINTER

__all__ = (
    'DLL', 'dlclose', 'CFUNC',
    'time_t', 'suseconds_t', 'timeval',
    'SOCKET', 'INVALID_SOCKET', 'socklen_t', 'sa_family_t', 'in_addr_t', 'in_port_t',
    'sockaddr', 'in_addr', 'sockaddr_in', 'in6_addr', 'sockaddr_in6',
    'FD_SETSIZE', 'fd_set', 'FD_ZERO', 'FD_ISSET', 'FD_SET', 'FD_CLR', 'select',
)

# X32 kernel interface is 64-bit.
if False:  # if defined __x86_64__ && defined __ILP32__
    # quad_t is also 64 bits.
    time_t = suseconds_t = ct.c_longlong
else:
    time_t = suseconds_t = ct.c_long
# endif

# Taken from the file <sys/time.h>
# #include <time.h>
#
# struct timeval {
#     time_t      tv_sec;   /* Seconds. */
#     suseconds_t tv_usec;  /* Microseconds. */
# };

class timeval(ct.Structure):
    _fields_ = [
    ("tv_sec",  time_t),       # seconds
    ("tv_usec", suseconds_t),  # microseconds
]

# Taken from the file libpcap's "socket.h"

# Some minor differences between sockets on various platforms.
# We include whatever sockets are needed for Internet-protocol
# socket access.

# In UN*X, a socket handle is a file descriptor, and therefore
# a signed integer.
SOCKET = ct.c_int

# In UN*X, the error return if socket() fails is -1.
INVALID_SOCKET = SOCKET(-1).value

class sockaddr(ct.Structure):
    _fields_ = [
    ("sa_family", ct.c_short),
    ("__pad1",    ct.c_ushort),
    ("ipv4_addr", ct.c_byte * 4),
    ("ipv6_addr", ct.c_byte * 16),
    ("__pad2",    ct.c_ulong),
]

# https://yarchive.net/comp/linux/socklen_t.html
socklen_t = ct.c_uint32 if ct.sizeof(ct.c_uint) < 4 else ct.c_uint

# POSIX.1g specifies this type name for the `sa_family' member.
sa_family_t = ct.c_ushort

# Types to represent an address and port.
in_addr_t = ct.c_uint32
in_port_t = ct.c_uint16

# IPv4 AF_INET sockets:

class in_addr(ct.Structure):
    _fields_ = [
    ("s_addr", in_addr_t),
]

class sockaddr_in(ct.Structure):
    _fields_ = [
    ("sin_family", sa_family_t),      # e.g. AF_INET, AF_INET6
    ("sin_port",   in_port_t),        # Port number.
    ("sin_addr",   in_addr),          # Internet address.
    ("sin_zero",   (ct.c_ubyte        # Pad to size of `struct sockaddr'.
                    * (ct.sizeof(sockaddr)
                       - ct.sizeof(sa_family_t)
                       - ct.sizeof(in_port_t)
                       - ct.sizeof(in_addr)))),
]

# IPv6 AF_INET6 sockets:

class in6_addr(ct.Union):
    _fields_ = [
    ("s6_addr",   (ct.c_uint8 * 16)),
    ("s6_addr16", (ct.c_uint16 * 8)),
    ("s6_addr32", (ct.c_uint32 * 4)),
]

class sockaddr_in6(ct.Structure):
    _fields_ = [
    ("sin6_family",   sa_family_t),   # address family, AF_INET6
    ("sin6_port",     in_port_t),     # Transport layer port #
    ("sin6_flowinfo", ct.c_uint32),   # IPv6 flow information
    ("sin6_addr",     in6_addr),      # IPv6 address
    ("sin6_scope_id", ct.c_uint32),   # IPv6 scope-id
]

# From <sys/select.h>

# The fd_set member
_fd_mask = ct.c_long  # fd_mask and NFDBITS are not portable, cannot be exported.
_NFDBITS = 8 * ct.sizeof(_fd_mask)

# Maximum number of file descriptors in `fd_set'.
FD_SETSIZE = 1024

class fd_set(ct.Structure):
    _fields_ = [
    ("fds_bits", (_fd_mask * (FD_SETSIZE // _NFDBITS))),
]

def FD_ZERO(fdsetp: POINTER[fd_set]) -> None:
    import ctypes as ct
    fdset = fdsetp.contents
    ct.memset(fdsetp, 0, ct.sizeof(fdset))
FD_ZERO = CFUNC(None, ct.POINTER(fd_set))(FD_ZERO)

def FD_ISSET(fd: int, fdsetp: POINTER[fd_set]) -> int:
    fdset = fdsetp.contents
    return int(fdset.fds_bits[fd // _NFDBITS] & (1 << (fd % _NFDBITS)))
FD_ISSET = CFUNC(ct.c_int, ct.c_int, ct.POINTER(fd_set))(FD_ISSET)

def FD_SET(fd: int, fdsetp: POINTER[fd_set]) -> None:
    fdset = fdsetp.contents
    fdset.fds_bits[fd // _NFDBITS] |= (1 << (fd % _NFDBITS))
FD_SET = CFUNC(None, ct.c_int, ct.POINTER(fd_set))(FD_SET)

def FD_CLR(fd: int, fdsetp: POINTER[fd_set]) -> None:
    fdset = fdsetp.contents
    fdset.fds_bits[fd // _NFDBITS] &= ~(1 << (fd % _NFDBITS))
FD_CLR = CFUNC(None, ct.c_int, ct.POINTER(fd_set))(FD_CLR)

libc = ct.CDLL("libc.so.6")

select = libc.select
select.restype  = ct.c_int
select.argtypes = [ct.c_int,
                   ct.POINTER(fd_set), ct.POINTER(fd_set), ct.POINTER(fd_set),
                   ct.POINTER(timeval)]

del ct, POINTER, libc
