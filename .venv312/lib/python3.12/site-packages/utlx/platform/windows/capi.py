# flake8-in-file-ignores: noqa: E305,F401,N813,N814,F821

# Copyright (c) 1994 Adam Karpierz
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import ctypes as ct
from ctypes import WinDLL as DLL
try:
    from _ctypes import FreeLibrary as dlclose  # type: ignore[attr-defined,unused-ignore]
except ImportError:  # pragma: no cover
    dlclose = lambda handle: None
from ctypes import WINFUNCTYPE as CFUNC
from ctypes import windll

from ...ctypes import POINTER

__all__ = (
    'DLL', 'dlclose', 'CFUNC',
    'time_t', 'suseconds_t', 'timeval',
    'SOCKET', 'INVALID_SOCKET', 'socklen_t', 'sa_family_t', 'in_addr_t', 'in_port_t',
    'sockaddr', 'in_addr', 'sockaddr_in', 'in6_addr', 'sockaddr_in6',
    'FD_SETSIZE', 'fd_set', 'FD_ZERO', 'FD_ISSET', 'FD_SET', 'FD_CLR', 'select',
)

time_t = ct.c_uint64

# Winsock doesn't have this POSIX type; it's used for the
# tv_usec value of struct timeval.
suseconds_t = ct.c_long

# Taken from the file <winsock.h>
#
# struct timeval {
#     long tv_sec;   /* seconds */
#     long tv_usec;  /* and microseconds */
# };

class timeval(ct.Structure):
    _fields_ = [
    ("tv_sec",  ct.c_long),    # seconds
    ("tv_usec", suseconds_t),  # microseconds
]

# Taken from the file libpcap's "socket.h"

# Some minor differences between sockets on various platforms.
# We include whatever sockets are needed for Internet-protocol
# socket access.

# In Winsock, a socket handle is of type SOCKET.
SOCKET = ct.c_uint

# In Winsock, the error return if socket() fails is INVALID_SOCKET.
INVALID_SOCKET = SOCKET(-1).value

# Winsock doesn't have this UN*X type; it's used in the UN*X sockets API.
socklen_t = ct.c_int

class sockaddr(ct.Structure):
    _fields_ = [
    ("sa_family", ct.c_short),
    ("__pad1",    ct.c_ushort),
    ("ipv4_addr", ct.c_byte * 4),
    ("ipv6_addr", ct.c_byte * 16),
    ("__pad2",    ct.c_ulong),
]

# POSIX.1g specifies this type name for the `sa_family' member.
sa_family_t = ct.c_short

# Types to represent an address and port.
in_addr_t = ct.c_uint32  # ct.c_ulong
in_port_t = ct.c_ushort

# IPv4 AF_INET sockets:

class in_addr(ct.Structure):
    _fields_ = [
    ("s_addr", in_addr_t),
]

class sockaddr_in(ct.Structure):
    _fields_ = [
    ("sin_family", sa_family_t),      # e.g. AF_INET, AF_INET6
    ("sin_port",   in_port_t),        # e.g. htons(3490)
    ("sin_addr",   in_addr),          # see struct in_addr, above
    ("sin_zero",   (ct.c_char * 8)),  # padding, zero this if you want to
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
    ("sin6_port",     in_port_t),     # port number, Network Byte Order
    ("sin6_flowinfo", ct.c_ulong),    # IPv6 flow information
    ("sin6_addr",     in6_addr),      # IPv6 address
    ("sin6_scope_id", ct.c_ulong),    # Scope ID
]

# From <sys/select.h>

# Maximum number of file descriptors in `fd_set'.
FD_SETSIZE = 1024

class fd_set(ct.Structure):
    _fields_ = [
    ("fd_count", ct.c_uint),
    ("fd_array", (SOCKET * FD_SETSIZE)),
]

def FD_ZERO(fdsetp: POINTER[fd_set]) -> None:
    import ctypes as ct
    fdset = fdsetp.contents
    ct.memset(fdsetp, 0, ct.sizeof(fdset))
FD_ZERO = CFUNC(None, ct.POINTER(fd_set))(FD_ZERO)

def FD_ISSET(fd: int, fdsetp: POINTER[fd_set]) -> int:
    fdset = fdsetp.contents
    for i in range(fdset.fd_count):
        if fdset.fd_array[i] == fd:
            return 1
    return 0
FD_ISSET = CFUNC(ct.c_int, ct.c_int, ct.POINTER(fd_set))(FD_ISSET)

def FD_SET(fd: int, fdsetp: POINTER[fd_set]) -> None:
    fdset = fdsetp.contents
    if fdset.fd_count < FD_SETSIZE:
        fdset.fd_array[fdset.fd_count] = fd
        fdset.fd_count += 1
    else: pass  # pragma: no cover
FD_SET = CFUNC(None, ct.c_int, ct.POINTER(fd_set))(FD_SET)

def FD_CLR(fd: int, fdsetp: POINTER[fd_set]) -> None:
    fdset = fdsetp.contents
    for i in range(fdset.fd_count):
        if fdset.fd_array[i] == fd:
            for j in range(i, fdset.fd_count - 1):
                fdset.fd_array[j] = fdset.fd_array[j + 1]
            fdset.fd_array[fdset.fd_count - 1] = 0
            fdset.fd_count -= 1
            break
FD_CLR = CFUNC(None, ct.c_int, ct.POINTER(fd_set))(FD_CLR)

select = windll.Ws2_32.select
select.restype  = ct.c_int
select.argtypes = [ct.c_int,
                   ct.POINTER(fd_set), ct.POINTER(fd_set), ct.POINTER(fd_set),
                   ct.POINTER(timeval)]

del ct, POINTER, windll
