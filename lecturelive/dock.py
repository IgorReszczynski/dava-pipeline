"""Dock a window to the right edge of its monitor, top to bottom, as a Windows app bar.

An app bar reserves its strip of the screen the way the taskbar does: maximized windows
resize to the space that is left instead of covering the panel. The shell decides the
final rectangle (it keeps the taskbar visible) and tells us when it changes.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

shell32 = ctypes.windll.shell32
user32 = ctypes.windll.user32

ABM_NEW, ABM_REMOVE, ABM_QUERYPOS, ABM_SETPOS = 0, 1, 2, 3
ABN_POSCHANGED = 1
ABE_RIGHT = 2
MONITOR_DEFAULTTONEAREST = 2
SWP_NOZORDER, SWP_NOACTIVATE = 0x0004, 0x0010
CALLBACK_MESSAGE = 0x8000 + 0x51  # WM_APP + n


class APPBARDATA(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("hWnd", wintypes.HWND), ("uCallbackMessage", wintypes.UINT),
                ("uEdge", wintypes.UINT), ("rc", wintypes.RECT), ("lParam", wintypes.LPARAM)]


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD)]


shell32.SHAppBarMessage.argtypes = [wintypes.DWORD, ctypes.POINTER(APPBARDATA)]
shell32.SHAppBarMessage.restype = ctypes.c_size_t
user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
user32.MonitorFromWindow.restype = wintypes.HMONITOR
user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MONITORINFO)]
user32.GetMonitorInfoW.restype = wintypes.BOOL
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, wintypes.UINT]
user32.SetWindowPos.restype = wintypes.BOOL


class AppBar:
    def __init__(self, hwnd: int, width_fraction: float):
        self.hwnd = hwnd
        self.fraction = width_fraction
        self.registered = False

    def _data(self) -> APPBARDATA:
        abd = APPBARDATA()
        abd.cbSize = ctypes.sizeof(APPBARDATA)
        abd.hWnd = self.hwnd
        abd.uCallbackMessage = CALLBACK_MESSAGE
        abd.uEdge = ABE_RIGHT
        return abd

    def register(self):
        if self.registered:
            return
        abd = self._data()
        if not shell32.SHAppBarMessage(ABM_NEW, ctypes.byref(abd)):
            raise OSError("Windows refused to dock the window to the right edge of the screen (ABM_NEW failed).")
        self.registered = True
        self.place()

    def place(self):
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        if not user32.GetMonitorInfoW(user32.MonitorFromWindow(self.hwnd, MONITOR_DEFAULTTONEAREST), ctypes.byref(mi)):
            raise ctypes.WinError()
        m = mi.rcMonitor
        width = round((m.right - m.left) * self.fraction)
        abd = self._data()
        abd.rc = wintypes.RECT(m.right - width, m.top, m.right, m.bottom)
        shell32.SHAppBarMessage(ABM_QUERYPOS, ctypes.byref(abd))   # the shell trims it around the taskbar
        abd.rc.left = abd.rc.right - width
        shell32.SHAppBarMessage(ABM_SETPOS, ctypes.byref(abd))
        r = abd.rc
        if not user32.SetWindowPos(self.hwnd, None, r.left, r.top, r.right - r.left, r.bottom - r.top,
                                   SWP_NOZORDER | SWP_NOACTIVATE):
            raise ctypes.WinError()

    def remove(self):
        if self.registered:
            shell32.SHAppBarMessage(ABM_REMOVE, ctypes.byref(self._data()))
            self.registered = False

    def handle(self, message_address: int) -> bool:
        """Call from nativeEvent. True if the message was the shell telling us to re-dock."""
        msg = wintypes.MSG.from_address(message_address)
        if msg.message == CALLBACK_MESSAGE:
            if msg.wParam == ABN_POSCHANGED and self.registered:
                self.place()
            return True
        return False
