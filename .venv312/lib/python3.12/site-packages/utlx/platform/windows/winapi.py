# flake8-in-file-ignores: noqa: E305,F401

# Copyright (c) 1994 Adam Karpierz
# SPDX-License-Identifier: Zlib

import ctypes
from ctypes import windll
from ctypes import wintypes
from ctypes import WINFUNCTYPE
from ctypes.wintypes import (
    CHAR, WCHAR, BOOLEAN, BOOL, BYTE, WORD, DWORD, SHORT, USHORT, INT,
    UINT, LONG, ULONG, LARGE_INTEGER, ULARGE_INTEGER, FLOAT, DOUBLE,
    LPBYTE, PBYTE, LPWORD, PWORD, LPDWORD, PDWORD, LPLONG, PLONG, LPSTR,
    LPWSTR, LPCSTR, LPVOID, LPCVOID, LPVOID as PVOID, HANDLE, LPHANDLE,
    PHANDLE, HMODULE, HLOCAL, WPARAM, LPARAM, FILETIME, LPFILETIME,
)

from ctypes.wintypes import WPARAM as ULONG_PTR  # workaround
PULONG_PTR = ctypes.POINTER(ULONG_PTR)

ULONG32   = ctypes.c_uint32
ULONGLONG = ctypes.c_uint64
DWORDLONG = ctypes.c_uint64
SIZE_T    = ctypes.c_size_t

WAIT_ABANDONED = 0x00000080
WAIT_OBJECT_0  = 0x00000000
WAIT_TIMEOUT   = 0x00000102
WAIT_FAILED    = 0xFFFFFFFF

IGNORE   = 0
INFINITE = 0xFFFFFFFF

FORMAT_MESSAGE_ALLOCATE_BUFFER = 0x00000100
FORMAT_MESSAGE_ARGUMENT_ARRAY  = 0x00002000
FORMAT_MESSAGE_FROM_HMODULE    = 0x00000800
FORMAT_MESSAGE_FROM_STRING     = 0x00000400
FORMAT_MESSAGE_FROM_SYSTEM     = 0x00001000
FORMAT_MESSAGE_IGNORE_INSERTS  = 0x00000200
FORMAT_MESSAGE_MAX_WIDTH_MASK  = 0x000000FF

GetActiveProcessorCount = windll.kernel32.GetActiveProcessorCount
GetActiveProcessorCount.restype  = DWORD
GetActiveProcessorCount.argtypes = [WORD]

GetModuleFileNameW = windll.kernel32.GetModuleFileNameW
GetModuleFileNameW.restype  = DWORD
GetModuleFileNameW.argtypes = [HMODULE,
                               LPWSTR,
                               DWORD]

GetProcessId = windll.kernel32.GetProcessId
GetProcessId.restype  = DWORD
GetProcessId.argtypes = [HANDLE]

GetCurrentProcess = windll.kernel32.GetCurrentProcess
GetCurrentProcess.restype  = HANDLE
GetCurrentProcess.argtypes = []

GetCurrentProcessId = windll.kernel32.GetCurrentProcessId
GetCurrentProcessId.restype  = DWORD
GetCurrentProcessId.argtypes = []

GetProcessTimes = windll.kernel32.GetProcessTimes
GetProcessTimes.restype  = BOOL
GetProcessTimes.argtypes = [HANDLE,
                            LPFILETIME,
                            LPFILETIME,
                            LPFILETIME,
                            LPFILETIME]

OpenProcessToken = windll.advapi32.OpenProcessToken
OpenProcessToken.restype  = BOOL
OpenProcessToken.argtypes = [HANDLE,   # ProcessHandle
                             DWORD,    # DesiredAccess
                             PHANDLE]  # TokenHandle

TOKEN_INFORMATION_CLASS  = DWORD
PTOKEN_INFORMATION_CLASS = PDWORD

GetTokenInformation = windll.advapi32.GetTokenInformation
GetTokenInformation.restype  = BOOL
GetTokenInformation.argtypes = [HANDLE,  # TokenHandle
                                TOKEN_INFORMATION_CLASS, # TokenInformationClass
                                LPVOID,  # TokenInformation
                                DWORD,   # TokenInformationLength
                                PDWORD]  # ReturnLength

GetExitCodeProcess = windll.kernel32.GetExitCodeProcess
GetExitCodeProcess.restype  = BOOL
GetExitCodeProcess.argtypes = [HANDLE, LPDWORD]

ExitProcess = windll.kernel32.ExitProcess
ExitProcess.restype  = None
ExitProcess.argtypes = [UINT]

class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
    ("nLength",              DWORD),
    ("lpSecurityDescriptor", LPVOID),
    ("bInheritHandle",       BOOL),
]
LPSECURITY_ATTRIBUTES = ctypes.POINTER(SECURITY_ATTRIBUTES)

LPTHREAD_START_ROUTINE = WINFUNCTYPE(DWORD, LPVOID)
CreateThread = windll.kernel32.CreateThread
CreateThread.restype  = HANDLE
CreateThread.argtypes = [LPSECURITY_ATTRIBUTES,
                         SIZE_T,
                         LPTHREAD_START_ROUTINE,
                         LPVOID,
                         DWORD,
                         LPDWORD]

GetCurrentThreadId = windll.kernel32.GetCurrentThreadId
GetCurrentThreadId.restype  = DWORD
GetCurrentThreadId.argtypes = []

WaitForSingleObject = windll.kernel32.WaitForSingleObject
WaitForSingleObject.restype  = DWORD
WaitForSingleObject.argtypes = [HANDLE,
                                DWORD]

SetEvent = windll.kernel32.SetEvent
SetEvent.restype  = BOOL
SetEvent.argtypes = [HANDLE]

CreateSemaphore = windll.kernel32.CreateSemaphoreA
CreateSemaphore.restype  = HANDLE
CreateSemaphore.argtypes = [LPSECURITY_ATTRIBUTES,
                            LONG,
                            LONG,
                            LPCSTR]

ReleaseSemaphore = windll.kernel32.ReleaseSemaphore
ReleaseSemaphore.restype  = BOOL
ReleaseSemaphore.argtypes = [HANDLE,
                             LONG,
                             LPLONG]

Sleep = windll.kernel32.Sleep
Sleep.restype  = None
Sleep.argtypes = [DWORD]

GetStdHandle = windll.kernel32.GetStdHandle
GetStdHandle.restype  = HANDLE
GetStdHandle.argtypes = [DWORD]

DuplicateHandle = windll.kernel32.DuplicateHandle
DuplicateHandle.restype  = BOOL
DuplicateHandle.argtypes = [HANDLE,
                            HANDLE,
                            HANDLE,
                            LPHANDLE,
                            DWORD,
                            BOOL,
                            DWORD]

SetHandleInformation = windll.kernel32.SetHandleInformation
SetHandleInformation.restype  = BOOL
SetHandleInformation.argtypes = [HANDLE,
                                 DWORD,
                                 DWORD]

CloseHandle = windll.kernel32.CloseHandle
CloseHandle.restype  = BOOL
CloseHandle.argtypes = [HANDLE]

# DWORD WINAPI GetLastError(void);
GetLastError = windll.kernel32.GetLastError
GetLastError.restype  = DWORD
GetLastError.argtypes = []

FormatMessageA = windll.kernel32.FormatMessageA
FormatMessageA.restype  = DWORD
FormatMessageA.argtypes = [DWORD,
                           LPCVOID,
                           DWORD,
                           DWORD,
                           LPSTR,
                           DWORD,
                           LPVOID]

class SYSTEMTIME(ctypes.Structure):
    _fields_ = [
    ("wYear",         WORD),
    ("wMonth",        WORD),
    ("wDayOfWeek",    WORD),
    ("wDay",          WORD),
    ("wHour",         WORD),
    ("wMinute",       WORD),
    ("wSecond",       WORD),
    ("wMilliseconds", WORD),
]
LPSYSTEMTIME = ctypes.POINTER(SYSTEMTIME)

GetLocalTime = windll.kernel32.GetLocalTime
GetLocalTime.restype  = None
GetLocalTime.argtypes = [LPSYSTEMTIME]

SetLocalTime = windll.kernel32.SetLocalTime
SetLocalTime.restype  = BOOL
SetLocalTime.argtypes = [LPSYSTEMTIME]

GetSystemTime = windll.kernel32.GetSystemTime
GetSystemTime.restype  = None
GetSystemTime.argtypes = [LPSYSTEMTIME]

SetSystemTime = windll.kernel32.SetSystemTime
SetSystemTime.restype  = BOOL
SetSystemTime.argtypes = [LPSYSTEMTIME]

# http://posted-stuff.blogspot.com/2009/07/iocp-based-sockets-with-ctypes-in_22.html

WSADESCRIPTION_LEN = 256
WSASYS_STATUS_LEN  = 128

class WSADATA(ctypes.Structure):
    _fields_ = [
    ("wVersion",       WORD),
    ("wHighVersion",   WORD),
    ("szDescription",  ctypes.c_char * (WSADESCRIPTION_LEN + 1)),
    ("szSystemStatus", ctypes.c_char * (WSASYS_STATUS_LEN  + 1)),
    ("iMaxSockets",    ctypes.c_ushort),
    ("iMaxUdpDg",      ctypes.c_ushort),
    ("lpVendorInfo",   ctypes.c_char_p),
]

LPWSADATA = ctypes.POINTER(WSADATA)

WSAStartup = windll.Ws2_32.WSAStartup
WSAStartup.restype  = ctypes.c_int
WSAStartup.argtypes = [WORD, LPWSADATA]

WSACleanup = windll.Ws2_32.WSACleanup
WSACleanup.restype  = ctypes.c_int
WSACleanup.argtypes = []

CP_ACP        = 0      # default to ANSI code page
CP_OEMCP      = 1      # default to OEM  code page
CP_MACCP      = 2      # default to MAC  code page
CP_THREAD_ACP = 3      # current thread's ANSI code page
CP_SYMBOL     = 42     # SYMBOL translations

CP_UTF7       = 65000  # UTF-7 translation
CP_UTF8       = 65001  # UTF-8 translation

# UINT WINAPI GetConsoleCP(void);
GetConsoleCP = windll.kernel32.GetConsoleCP
GetConsoleCP.restype  = UINT
GetConsoleCP.argtypes = []

# UINT WINAPI GetConsoleOutputCP(void);
GetConsoleOutputCP = windll.kernel32.GetConsoleOutputCP
GetConsoleOutputCP.restype  = UINT
GetConsoleOutputCP.argtypes = []

# BOOL WINAPI SetConsoleCP(
#  _In_  UINT wCodePageID
# );
SetConsoleCP = windll.kernel32.SetConsoleCP
SetConsoleCP.restype  = BOOL
SetConsoleCP.argtypes = [UINT]

# BOOL WINAPI SetConsoleOutputCP(
#  _In_  UINT wCodePageID
# );
SetConsoleOutputCP = windll.kernel32.SetConsoleOutputCP
SetConsoleOutputCP.restype  = BOOL
SetConsoleOutputCP.argtypes = [UINT]

PHANDLER_ROUTINE = WINFUNCTYPE(BOOL, DWORD)
SetConsoleCtrlHandler = windll.kernel32.SetConsoleCtrlHandler
SetConsoleCtrlHandler.restype  = BOOL
SetConsoleCtrlHandler.argtypes = [PHANDLER_ROUTINE,
                                  BOOL]

LocalFree = windll.kernel32.LocalFree
LocalFree.restype  = HLOCAL
LocalFree.argtypes = [HLOCAL]

MAKEWORD = lambda blow, bhigh: (bhigh << 8) + blow

del ctypes
