"""The window: a translucent panel docked to the right edge of the screen, top to bottom.

The design follows Material 3 Expressive: the M3 shape scale with the Expressive radii
(large-increased 20, extra-large-increased 32, extra-extra-large 48, full), tonal surfaces
from the M3 baseline scheme, a record button that morphs from a circle to a rounded square,
M3 switches, an Expressive slider and a connected button group.

The app opens on the module picker (which course is this lecture for); each module has its
own folder. The recording view stays minimal: record, level, transcript. Everything else
sits in a settings sheet, and exports in a menu.
"""
from __future__ import annotations

import atexit
import math
import sys
import threading
import time
import traceback
from pathlib import Path

from PySide6.QtCore import (QEasingCurve, QEvent, QObject, QPoint, QPointF, QPropertyAnimation, QRectF, Qt, QTimer,
                            QUrl, QVariantAnimation, Signal)
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QGuiApplication, QIcon, QKeySequence, QPainter,
                           QPainterPath, QPalette, QPen, QPixmap, QRadialGradient, QShortcut, QTextBlockFormat,
                           QTextCharFormat, QTextCursor)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QButtonGroup, QComboBox, QDialog, QFileDialog, QFrame,
                               QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QLineEdit, QListView, QMenu,
                               QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QStackedWidget, QTextEdit, QVBoxLayout, QWidget)

from . import config, downloads, export, notes
from .audio import InputDevice, list_input_devices, rescan
from .dock import AppBar
from .engine import language_codes
from .live import PROFILES, Final, Level, Live, Partial, Status, hms, line
from .modules import Module, ModuleStore, check_name

APP = "Lecture Live"

# Material 3 shape scale in dp (= Qt logical px), with the Expressive additions.
SHAPE = dict(xs=4, sm=8, md=12, lg=16, lg_inc=20, xl=28, xl_inc=32, xxl=48)

# Material 3 baseline color scheme.
SCHEMES = {
    "light": dict(
        primary="#6750A4", on_primary="#FFFFFF", primary_container="#EADDFF", on_primary_container="#21005D",
        secondary="#625B71", secondary_container="#E8DEF8", on_secondary_container="#1D192B", tertiary="#7D5260",
        error="#B3261E", on_error="#FFFFFF",
        surface="#FEF7FF", surface_container_low="#F7F2FA", surface_container="#F3EDF7",
        surface_container_high="#ECE6F0", surface_container_highest="#E6E0E9",
        on_surface="#1D1B20", on_surface_variant="#49454F", outline="#79747E", outline_variant="#CAC4D0"),
    "dark": dict(
        primary="#D0BCFF", on_primary="#381E72", primary_container="#4F378B", on_primary_container="#EADDFF",
        secondary="#CCC2DC", secondary_container="#4A4458", on_secondary_container="#E8DEF8", tertiary="#EFB8C8",
        error="#F2B8B5", on_error="#601410",
        surface="#141218", surface_container_low="#1D1B20", surface_container="#211F26",
        surface_container_high="#2B2930", surface_container_highest="#36343B",
        on_surface="#E6E0E9", on_surface_variant="#CAC4D0", outline="#938F99", outline_variant="#49454F"),
}

TEXT_FONT = "Segoe UI Variable Text"
DISPLAY_FONT = "Segoe UI Variable Display"
ICON_FONT = "Segoe Fluent Icons"
ICONS = dict(minimize="", close="", settings="", copy="", folder="",
             refresh="", export="", remove="", chevron="", add="", note="")
SPEAKER_COLORS = ("primary", "tertiary", "secondary")

LANGUAGE_NAMES = {
    "en": "English", "de": "German", "pl": "Polish", "fr": "French", "es": "Spanish", "it": "Italian",
    "nl": "Dutch", "pt": "Portuguese", "uk": "Ukrainian", "ru": "Russian", "cs": "Czech", "sk": "Slovak",
    "sv": "Swedish", "da": "Danish", "no": "Norwegian", "fi": "Finnish", "tr": "Turkish", "el": "Greek",
    "hu": "Hungarian", "ro": "Romanian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "hi": "Hindi", "he": "Hebrew", "vi": "Vietnamese", "id": "Indonesian",
}


def pick_scheme() -> dict:
    dark = QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
    return SCHEMES["dark" if dark else "light"]


def rgba(hex_color: str, alpha: float) -> str:
    c = QColor(hex_color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {round(alpha * 255)})"


def qcolor(hex_color: str, alpha: float = 1.0) -> QColor:
    c = QColor(hex_color)
    c.setAlphaF(alpha)
    return c


def font(px: float, weight=QFont.Weight.Normal, family: str = TEXT_FONT, tabular: bool = False) -> QFont:
    f = QFont(family)
    f.setPixelSize(round(px))
    f.setWeight(weight)
    if tabular:
        f.setFeature(QFont.Tag("tnum"), 1)
    return f


def glyph_icon(glyph: str, color: str, px: int) -> QIcon:
    """A Segoe Fluent Icons glyph as a QIcon (for buttons whose text uses the text font)."""
    scale = 3
    pm = QPixmap(px * scale, px * scale)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QColor(color))
    p.setFont(font(px * scale, family=ICON_FONT))
    p.drawText(pm.rect(), Qt.AlignCenter, glyph)
    p.end()
    pm.setDevicePixelRatio(scale)
    return QIcon(pm)


def make_icon(size: int = 256) -> QPixmap:
    """App icon: a rounded square (Expressive xl corner ratio) with a waveform."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#6750A4"))
    p.drawRoundedRect(QRectF(0, 0, size, size), size * 0.3, size * 0.3)
    p.setBrush(QColor("#EADDFF"))
    bars = [0.28, 0.52, 0.78, 0.46, 0.64, 0.34]
    w, gap = size * 0.075, size * 0.045
    x = (size - (len(bars) * w + (len(bars) - 1) * gap)) / 2
    for h in bars:
        bh = size * 0.62 * h
        p.drawRoundedRect(QRectF(x, (size - bh) / 2, w, bh), w / 2, w / 2)
        x += w + gap
    p.end()
    return pm


class Bridge(QObject):
    event = Signal(str, object)


# ---------------------------------------------------------------------------------------------- widgets
def morph_path(cx: float, cy: float, radius: float, t: float, rotation: float, lobes: int = 9,
               depth: float = 0.075, points: int = 180) -> QPainterPath:
    """A shape between a scalloped "cookie" (t=0) and a squircle (t=1), from the Material 3
    Expressive shape set, drawn in polar form so the two can be blended for a morph."""
    path = QPainterPath()
    for i in range(points + 1):
        a = 2 * math.pi * i / points
        cookie = radius * (1 + depth * math.cos(lobes * (a + rotation)))
        c, s = abs(math.cos(a)), abs(math.sin(a))
        squircle = radius * 0.92 / (c ** 5 + s ** 5) ** 0.2
        r = cookie + (squircle - cookie) * t
        pt = QPointF(cx + r * math.cos(a), cy + r * math.sin(a))
        path.moveTo(pt) if i == 0 else path.lineTo(pt)
    path.closeSubpath()
    return path


class RecordButton(QAbstractButton):
    """Material 3 Expressive record button. Idle: a softly scalloped cookie with a microphone.
    Recording: it morphs into a squircle with a stop sign, and a halo behind it turns and grows
    with the input level, so you can see the app hearing the room. Stopping: a spinner."""
    SIZE = 112          # widget; the shape is about 76 px across, the rest is room for the halo
    RADIUS = 36

    def __init__(self, t: dict):
        super().__init__()
        self.t = t
        self.mode = "idle"          # idle | recording | stopping
        self._t = 0.0               # 0 = cookie, 1 = squircle
        self._scale = 1.0
        self._spin = 0.0
        self._level = 0.0           # 0..1, smoothed input level
        self._target_level = 0.0
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover)
        self._morph = QVariantAnimation(self, duration=550, easingCurve=QEasingCurve.OutBack)
        self._morph.valueChanged.connect(lambda v: self._set("_t", v))
        self._squish = QVariantAnimation(self, duration=260, easingCurve=QEasingCurve.OutBack)
        self._squish.valueChanged.connect(lambda v: self._set("_scale", v))
        self._tick = QTimer(self, interval=33, timeout=self._advance)   # ~30 fps, only while it moves
        self.pressed.connect(lambda: self._squish_to(0.9))
        self.released.connect(lambda: self._squish_to(1.0))
        shadow = QGraphicsDropShadowEffect(self, blurRadius=28, offset=QPointF(0, 6))
        shadow.setColor(QColor(0, 0, 0, 90))
        self.setGraphicsEffect(shadow)

    def _set(self, name: str, v):
        setattr(self, name, float(v))
        self.update()

    def _squish_to(self, s: float):
        self._squish.stop()
        self._squish.setStartValue(self._scale)
        self._squish.setEndValue(s)
        self._squish.start()

    def set_mode(self, mode: str):
        self.mode = mode
        self._morph.stop()
        self._morph.setStartValue(self._t)
        self._morph.setEndValue(0.0 if mode == "idle" else 1.0)
        self._morph.start()
        self._update_ticking()

    def set_level(self, rms_db: float):
        """Input level in dBFS; -50 dB and below is still, -12 dB fills the halo."""
        db = rms_db if math.isfinite(rms_db) else -90.0
        self._target_level = max(0.0, min(1.0, (db + 50) / 38))

    def enterEvent(self, ev):
        self._update_ticking(hover=True)
        super().enterEvent(ev)

    def leaveEvent(self, ev):
        self._update_ticking(hover=False)
        super().leaveEvent(ev)

    def _update_ticking(self, hover: bool | None = None):
        moving = self.mode != "idle" or (self.underMouse() if hover is None else hover)
        if moving and not self._tick.isActive():
            self._tick.start()
        elif not moving:
            self._tick.stop()

    def _advance(self):
        speed = {"recording": 0.9, "stopping": 7.0}.get(self.mode, 0.35)   # degrees per frame
        self._spin = (self._spin + speed) % 360
        self._level += (self._target_level - self._level) * 0.25
        self.update()

    def paintEvent(self, _):
        t = self.t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx = cy = self.SIZE / 2
        R = self.RADIUS * self._scale
        rot = math.radians(self._spin)
        recording = self.mode == "recording" and self.isEnabled()
        p.setPen(Qt.NoPen)

        if recording or (self.underMouse() and self.isEnabled()):
            halo_color = "#E5484D" if recording else t["primary"]
            grow = 1.14 + (0.26 * self._level if recording else 0.04)
            alpha = (0.16 + 0.22 * self._level) if recording else 0.12
            p.setBrush(qcolor(halo_color, alpha))
            p.drawPath(morph_path(cx, cy, R * grow, 0.0, -rot * 0.6, lobes=12, depth=0.06))

        if not self.isEnabled():
            fill = qcolor(t["on_surface"], 0.12)
            fg = qcolor(t["on_surface"], 0.38)
        else:
            if recording:
                top, bottom, fg = QColor("#FF6B6F"), QColor("#C62A2F"), QColor("#FFFFFF")
            elif self.mode == "stopping":
                top = QColor(t["surface_container_highest"]).lighter(115)
                bottom, fg = QColor(t["surface_container_highest"]), QColor(t["primary"])
            else:
                base = QColor(t["primary"])
                top, bottom, fg = base.lighter(118), base.darker(112), QColor(t["on_primary"])
            fill = QRadialGradient(QPointF(cx - R * 0.45, cy - R * 0.55), R * 2.1)
            fill.setColorAt(0.0, top)
            fill.setColorAt(1.0, bottom)
        body = morph_path(cx, cy, R, self._t, rot if self.mode == "idle" else 0.0)
        p.setBrush(fill)
        p.drawPath(body)
        # a thin highlight along the top edge gives the surface some depth
        p.setPen(QPen(QColor(255, 255, 255, 46 if self.isEnabled() else 0), 1.2))
        p.setBrush(Qt.NoBrush)
        p.drawPath(body)
        if self.isDown():
            p.setPen(Qt.NoPen)
            p.setBrush(qcolor(fg.name(), 0.12))
            p.drawPath(body)

        p.setPen(Qt.NoPen)
        p.setBrush(fg)
        if self.mode == "recording":
            p.drawRoundedRect(QRectF(cx - 12, cy - 12, 24, 24), 7, 7)
        elif self.mode == "stopping":
            pen = QPen(fg)
            pen.setWidthF(4)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawArc(QRectF(cx - 14, cy - 14, 28, 28), int(-self._spin * 16), 270 * 16)
        else:
            p.setPen(fg)
            p.setFont(font(28 * self._scale, family=ICON_FONT))
            p.drawText(QRectF(cx - R, cy - R, 2 * R, 2 * R), Qt.AlignCenter, "")   # microphone
        p.end()


class Switch(QAbstractButton):
    """Material 3 switch: 52 x 32 track, thumb grows from 16 to 24 when on."""

    def __init__(self, t: dict, checked: bool):
        super().__init__()
        self.t = t
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(52, 32)
        self._pos = 1.0 if checked else 0.0
        self._anim = QVariantAnimation(self, duration=300, easingCurve=QEasingCurve.OutBack)
        self._anim.valueChanged.connect(self._set_pos)
        self.toggled.connect(self._on_toggled)

    def _set_pos(self, v):
        self._pos = float(v)
        self.update()

    def _on_toggled(self, on: bool):
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def paintEvent(self, _):
        t = self.t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        track = QRectF(1, 1, 50, 30)
        on = self.isChecked()
        if on:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(t["primary"]))
        else:
            p.setPen(QPen(QColor(t["outline"]), 2))
            p.setBrush(QColor(t["surface_container_highest"]))
        p.drawRoundedRect(track, 15, 15)
        k = max(0.0, min(1.0, self._pos))
        d = 28 if self.isDown() else 16 + 8 * k
        cx = 16 + (36 - 16) * self._pos
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t["on_primary"] if on else t["outline"]))
        p.drawEllipse(QPointF(cx, 16), d / 2, d / 2)
        p.end()


class IconButton(QPushButton):
    def __init__(self, glyph: str, tip: str):
        super().__init__(glyph)
        self.setObjectName("icon")
        self.setToolTip(tip)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(font(16, family=ICON_FONT))


class Select(QComboBox):
    """Filled tonal dropdown with a chevron (the stylesheet hides Qt's own arrow box)."""

    def __init__(self, t: dict):
        super().__init__()
        self.t = t
        self.setView(QListView())
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QPainter(self)
        p.setFont(font(12, family=ICON_FONT))
        p.setPen(QColor(self.t["on_surface_variant"] if self.isEnabled() else self.t["outline_variant"]))
        p.drawText(QRectF(self.width() - 34, 0, 22, self.height()), Qt.AlignCenter, "")
        p.end()


class SegmentedButtons(QWidget):
    """Connected button group: outer corners full, inner corners small; the selected
    segment morphs to fully rounded (Material 3 Expressive)."""
    changed = Signal(str)

    def __init__(self, t: dict, options: list[tuple[str, str, str]], value: str):
        super().__init__()
        self.t = t
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.group = QButtonGroup(self)
        self.buttons: list[tuple[QPushButton, str]] = []
        for label, data, tip in options:
            b = QPushButton(label)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
            b.setMinimumHeight(40)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setChecked(data == value)
            self.group.addButton(b)
            lay.addWidget(b)
            self.buttons.append((b, data))
        self.group.buttonToggled.connect(self._toggled)
        self._restyle()

    def value(self) -> str:
        return next(d for b, d in self.buttons if b.isChecked())

    def _toggled(self, _b, checked: bool):
        self._restyle()
        if checked:
            self.changed.emit(self.value())

    def _restyle(self):
        t, full, inner = self.t, 20, SHAPE["sm"]
        last = len(self.buttons) - 1
        for i, (b, _) in enumerate(self.buttons):
            sel = b.isChecked()
            left = full if (i == 0 or sel) else inner
            right = full if (i == last or sel) else inner
            bg = t["primary"] if sel else t["surface_container_highest"]
            fg = t["on_primary"] if sel else t["on_surface_variant"]
            b.setStyleSheet(
                f"QPushButton {{ background: {bg}; color: {fg}; border: none; font: 600 14px '{TEXT_FONT}';"
                f" border-top-left-radius: {left}px; border-bottom-left-radius: {left}px;"
                f" border-top-right-radius: {right}px; border-bottom-right-radius: {right}px; padding: 0 12px; }}"
                f"QPushButton:hover {{ background: {bg}; color: {fg if sel else t['on_surface']}; }}"
                f"QPushButton:disabled {{ color: {rgba(t['on_surface'], 0.38)}; }}")


class LevelMeter(QWidget):
    """A speech dot and a thin full-rounded level bar with peak hold."""
    LO, HI = -70.0, 0.0

    def __init__(self, t: dict, threshold: float):
        super().__init__()
        self.t = t
        self.threshold = threshold
        self.rms = self.LO
        self.hold = self.LO
        self.hold_at = 0.0
        self.clip_at = 0.0
        self.speech = False
        self.setFixedHeight(16)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip("Dot: speech heard (Silero VAD).\nBar: input level after gain. While the lecturer "
                        "talks it should fill past the middle and never turn red (clipping).")

    def set_level(self, lvl: Level):
        now = time.monotonic()
        db = max(self.LO, lvl.rms_db) if math.isfinite(lvl.rms_db) else self.LO
        self.rms = max(db, self.rms - 1.2)
        peak = max(self.LO, lvl.peak_db) if math.isfinite(lvl.peak_db) else self.LO
        if peak >= self.hold or now - self.hold_at > 1.5:
            self.hold, self.hold_at = peak, now
        if lvl.clipped:
            self.clip_at = now
        self.speech = lvl.speech_prob >= self.threshold
        self.update()

    def paintEvent(self, _):
        t = self.t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        h = self.height()
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t["primary"] if self.speech else t["outline_variant"]))
        p.drawEllipse(QPointF(5, h / 2), 5, 5)
        bar = QRectF(18, (h - 8) / 2, self.width() - 18, 8)
        p.setBrush(QColor(t["surface_container_highest"]))
        p.drawRoundedRect(bar, 4, 4)
        frac = (self.rms - self.LO) / (self.HI - self.LO)
        clipping = time.monotonic() - self.clip_at < 1.0
        color = t["error"] if clipping else t["outline"] if self.rms < -45 else t["primary"]
        p.setBrush(QColor(color))
        p.drawRoundedRect(QRectF(bar.left(), bar.top(), max(8.0, bar.width() * frac), 8), 4, 4)
        hx = bar.left() + bar.width() * (self.hold - self.LO) / (self.HI - self.LO)
        p.setBrush(QColor(t["on_surface_variant"]))
        p.drawRoundedRect(QRectF(min(hx, bar.right() - 3), bar.top() - 2, 3, 12), 1.5, 1.5)
        p.end()


class LogDialog(QDialog):
    def __init__(self, parent, lines: list[str]):
        super().__init__(parent)
        self.setWindowTitle(f"{APP} log")
        self.resize(900, 520)
        self.view = QPlainTextEdit(readOnly=True)
        self.view.setFont(QFont("Cascadia Mono", 9))
        self.view.setPlainText("\n".join(lines))
        lay = QVBoxLayout(self)
        lay.addWidget(self.view)

    def append(self, line: str):
        self.view.appendPlainText(line)


class ElidedLabel(QLabel):
    """A one-line label that elides in the middle (for folder paths)."""

    def __init__(self, text: str = ""):
        super().__init__()
        self._full = text
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setToolTip(text)

    def set_full_text(self, text: str):
        self._full = text
        self.setToolTip(text)
        self._elide()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._elide()

    def _elide(self):
        self.setText(self.fontMetrics().elidedText(self._full, Qt.ElideMiddle, max(0, self.width())))


class ModuleItem(QAbstractButton):
    """M3 list item on a tonal card: letter avatar, name, folder, remove button."""
    remove_requested = Signal(str)

    def __init__(self, t: dict, module: Module, current: bool, vault: Path | None):
        super().__init__()
        self.t = t
        self.module = module
        self.current = current
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover)
        self.setMinimumHeight(72)
        where = module.folder
        if module.vault_path is not None:
            rel = module.vault_folder
            if vault is not None and module.vault_path.is_relative_to(vault):
                rel = module.vault_path.relative_to(vault).as_posix()
            where = f"Obsidian · {rel}"
        self.setToolTip(f"Recordings: {module.folder}\nNotes: {module.vault_folder or 'none'}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(68, 10, 8, 10)
        texts = QVBoxLayout()
        texts.setSpacing(0)
        name = QLabel(module.name)
        name.setObjectName("rowTitle")
        folder = ElidedLabel(where)
        folder.setObjectName("rowSupport")
        for w in (name, folder):
            w.setAttribute(Qt.WA_TransparentForMouseEvents)
        texts.addWidget(name)
        texts.addWidget(folder)
        lay.addLayout(texts, 1)
        rm = IconButton(ICONS["remove"], "Remove from the list (the folder and its recordings stay on disk)")
        rm.clicked.connect(lambda: self.remove_requested.emit(module.name))
        lay.addWidget(rm, 0, Qt.AlignVCenter)

    def paintEvent(self, _):
        t = self.t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(t["secondary_container"] if self.current else t["surface_container_high"]))
        p.drawRoundedRect(r, SHAPE["lg_inc"], SHAPE["lg_inc"])
        if self.underMouse():
            p.setBrush(qcolor(t["on_surface"], 0.12 if self.isDown() else 0.08))
            p.drawRoundedRect(r, SHAPE["lg_inc"], SHAPE["lg_inc"])
        c = QPointF(36, r.height() / 2)
        p.setBrush(QColor(t["primary_container"]))
        p.drawEllipse(c, 20, 20)
        p.setPen(QColor(t["on_primary_container"]))
        p.setFont(font(16, QFont.Weight.DemiBold))
        p.drawText(QRectF(c.x() - 20, c.y() - 20, 40, 40), Qt.AlignCenter, self.module.name[:1].upper())
        p.end()


def caption(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("caption")
    return lbl


def setting_row(title: str, support: str, trailing: QWidget) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 2, 0, 2)
    texts = QVBoxLayout()
    texts.setSpacing(0)
    a = QLabel(title)
    a.setObjectName("rowTitle")
    b = QLabel(support)
    b.setObjectName("rowSupport")
    b.setWordWrap(True)
    texts.addWidget(a)
    texts.addWidget(b)
    lay.addLayout(texts, 1)
    lay.addWidget(trailing, 0, Qt.AlignVCenter)
    return w


# ---------------------------------------------------------------------------------------------- window
class MainWindow(QWidget):
    def __init__(self, cfg: config.Config, t: dict):
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint | Qt.WindowMinimizeButtonHint)
        self.cfg = cfg
        self.t = t
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setObjectName("root")
        self.setWindowTitle(APP)
        self.setWindowIcon(QIcon(make_icon()))
        self.bridge = Bridge()
        self.bridge.event.connect(self._on_event)
        self.live = Live(cfg, self.bridge.event.emit)
        self.dock: AppBar | None = None
        self.devices: list[InputDevice] = []
        self.log_lines: list[str] = []
        self.log_dialog: LogDialog | None = None
        self.state = "idle"
        self.engine_ok = False
        self.engine_error = False
        self.session_folder: Path | None = None
        self.started_at = 0.0
        self.next_final_id = 0
        self.partial_pos: int | None = None
        self.final_lines: list[str] = []
        self.overflows = 0
        self.quit_after_finish = False
        self.sheet_open = False
        self.store = ModuleStore(cfg.output_path)
        self.module: Module | None = None
        self.new_folder: Path | None = None    # folder chosen for a new module (None = default)
        self.new_vault_folder: Path | None = None
        self.note_path: Path | None = None
        self.note_busy = False
        self.last_speaker: int | None = None

        self._build()
        self._style()
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._toggle)
        self.clock = QTimer(self, interval=500, timeout=self._tick)
        self.clock.start()

        geo = QGuiApplication.primaryScreen().availableGeometry()
        width = round(QGuiApplication.primaryScreen().geometry().width() * cfg.panel_width)
        self.setGeometry(geo.right() + 1 - width, geo.top(), width, geo.height())
        QTimer.singleShot(0, self._startup)

    # ------------------------------------------------------------------ layout
    def _build(self):
        t = self.t
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 14, 14, 16)
        outer.setSpacing(14)

        top = QHBoxLayout()
        self.module_chip = QPushButton()
        self.module_chip.setObjectName("chip")
        self.module_chip.setCursor(Qt.PointingHandCursor)
        self.module_chip.setToolTip("Change module")
        self.module_chip.clicked.connect(self._show_picker)
        self.module_chip.setVisible(False)
        top.addWidget(self.module_chip)
        top.addStretch(1)
        mini = IconButton(ICONS["minimize"], "Minimize")
        mini.clicked.connect(self.showMinimized)
        close = IconButton(ICONS["close"], "Close")
        close.clicked.connect(self.close)
        top.addWidget(mini)
        top.addWidget(close)
        outer.addLayout(top)

        # page 0: module picker, page 1: recording view
        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)
        self.stack.addWidget(self._build_picker())
        main = QWidget()
        outer = QVBoxLayout(main)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(14)
        self.stack.addWidget(main)

        hero = QHBoxLayout()
        hero.setSpacing(16)
        self.rec_btn = RecordButton(t)
        self.rec_btn.setEnabled(False)
        self.rec_btn.setToolTip("Start or stop (Ctrl+R)")
        self.rec_btn.clicked.connect(self._toggle)
        hero.addWidget(self.rec_btn)
        texts = QVBoxLayout()
        texts.setSpacing(0)
        texts.addStretch(1)
        self.clock_lbl = QLabel("00:00:00")
        self.clock_lbl.setFont(font(36, QFont.Weight.Medium, DISPLAY_FONT, tabular=True))
        self.status_lbl = QLabel("Loading Whisper…")
        self.status_lbl.setObjectName("status")
        texts.addWidget(self.clock_lbl)
        texts.addWidget(self.status_lbl)
        texts.addStretch(1)
        hero.addLayout(texts, 1)
        outer.addLayout(hero)

        self.meter = LevelMeter(t, self.cfg.vad_threshold)
        outer.addWidget(self.meter)

        # settings sheet
        self.sheet = QFrame()
        self.sheet.setObjectName("sheet")
        sl = QVBoxLayout(self.sheet)
        sl.setContentsMargins(16, 14, 16, 16)
        sl.setSpacing(8)

        sl.addWidget(caption("Microphone"))
        mic_row = QHBoxLayout()
        self.dev_box = Select(t)
        self.dev_box.activated.connect(self._device_chosen)
        mic_row.addWidget(self.dev_box, 1)
        self.refresh_btn = IconButton(ICONS["refresh"], "Look for microphones again")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        mic_row.addWidget(self.refresh_btn)
        sl.addLayout(mic_row)

        sl.addWidget(caption("Language"))
        self.lang_box = Select(t)
        self._fill_languages()
        self.lang_box.currentIndexChanged.connect(self._lang_changed)
        sl.addWidget(self.lang_box)

        sl.addWidget(caption("Silence"))
        self.mode = SegmentedButtons(t, [
            ("Skip silence", "vad", "Silero VAD finds speech; a pause ends a chunk and silence is not sent."),
            ("All audio", "continuous", "Back-to-back chunks: every second of audio goes to Whisper."),
        ], self.cfg.segmentation)
        self.mode.changed.connect(self._mode_changed)
        sl.addWidget(self.mode)

        sl.addWidget(caption("GPU use"))
        self.profile = SegmentedButtons(t, [
            ("Accuracy", "accuracy", "Previews every 2 s with the full beam. The GPU is busy the whole lecture."),
            ("Balanced", "balanced", "Same final text; quicker previews every 4 s. About a third of the GPU."),
            ("Saver", "saver", "No live preview, faster decoding. About a tenth of the GPU; "
                               "text appears when each sentence ends."),
        ], self.cfg.compute_profile)
        self.profile.changed.connect(self._profile_changed)
        sl.addWidget(self.profile)

        sl.addWidget(caption("Gain"))
        gain_row = QHBoxLayout()
        self.gain = QSlider(Qt.Horizontal)
        self.gain.setRange(-12, 40)
        self.gain.setValue(round(self.cfg.input_gain_db))
        self.gain.setMinimumHeight(40)
        self.gain.valueChanged.connect(self._gain_changed)
        gain_row.addWidget(self.gain, 1)
        self.gain_lbl = QLabel()
        self.gain_lbl.setObjectName("value")
        self.gain_lbl.setFont(font(14, QFont.Weight.DemiBold, tabular=True))
        self.gain_lbl.setMinimumWidth(52)
        self.gain_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        gain_row.addWidget(self.gain_lbl)
        sl.addLayout(gain_row)
        self._gain_changed(self.gain.value())

        self.raw_sw = Switch(t, self.cfg.raw_capture)
        self.raw_sw.toggled.connect(self._raw_toggled)
        sl.addWidget(setting_row("Raw microphone", "Bypass Windows mic effects that treat a distant "
                                 "lecturer as noise", self.raw_sw))
        self.agc_sw = Switch(t, self.cfg.agc)
        self.agc_sw.toggled.connect(lambda v: setattr(self.live, "agc", v))
        sl.addWidget(setting_row("Auto level", f"Bring speech to {self.cfg.agc_target_dbfs:g} dBFS before "
                                 "Whisper", self.agc_sw))
        self.ctx_sw = Switch(t, self.cfg.use_context)
        self.ctx_sw.toggled.connect(lambda v: setattr(self.live, "use_context", v))
        sl.addWidget(setting_row("Transcript as context", "Keeps spelling consistent across chunks", self.ctx_sw))
        self.spk_sw = Switch(t, self.cfg.speakers)
        self.spk_sw.toggled.connect(lambda v: setattr(self.live, "speakers", v))
        sl.addWidget(setting_row("Tell speakers apart", "Labels each voice Speaker 1, 2, … (best effort "
                                 "from one microphone)", self.spk_sw))

        sl.addWidget(caption("Vocabulary"))
        self.prompt = QLineEdit(self.cfg.initial_prompt)
        self.prompt.setPlaceholderText("Course names and terms")
        self.prompt.textChanged.connect(lambda s: setattr(self.live, "prompt", s))
        sl.addWidget(self.prompt)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 6, 0, 0)
        self.hear_btn = QPushButton("Hear last 5 s")
        self.hear_btn.setObjectName("tonal")
        self.hear_btn.setToolTip("Play back what the app hears, after gain. Use headphones in class.")
        self.hear_btn.clicked.connect(self._hear)
        actions.addWidget(self.hear_btn)
        actions.addStretch(1)
        log_btn = QPushButton("Log")
        log_btn.setObjectName("text")
        log_btn.clicked.connect(self._show_log)
        actions.addWidget(log_btn)
        sl.addLayout(actions)

        self.sheet_wrap = QScrollArea()
        self.sheet_wrap.setObjectName("sheetWrap")
        self.sheet_wrap.setWidget(self.sheet)
        self.sheet_wrap.setWidgetResizable(True)
        self.sheet_wrap.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.sheet_wrap.setFrameShape(QFrame.NoFrame)
        self.sheet_wrap.setMaximumHeight(0)
        self.sheet_wrap.setVisible(False)
        self.sheet_anim = QPropertyAnimation(self.sheet_wrap, b"maximumHeight", self, duration=380,
                                             easingCurve=QEasingCurve.OutCubic)
        self.sheet_anim.finished.connect(self._sheet_anim_done)
        outer.addWidget(self.sheet_wrap)

        self.text = QTextEdit(readOnly=True)
        self.text.setObjectName("transcript")
        self.text.setPlaceholderText("Press record. The lecture appears here as it is spoken.")
        self.text.setFont(font(16))
        outer.addWidget(self.text, 1)

        foot = QHBoxLayout()
        foot.setSpacing(2)
        info = QVBoxLayout()
        info.setSpacing(0)
        self.stats_lbl = QLabel("")
        self.stats_lbl.setObjectName("stats")
        self.stats_lbl.setWordWrap(True)
        self.warn_lbl = QLabel("")
        self.warn_lbl.setObjectName("warn")
        self.warn_lbl.setWordWrap(True)
        self.warn_lbl.setVisible(False)
        info.addWidget(self.stats_lbl)
        info.addWidget(self.warn_lbl)
        foot.addLayout(info, 1)
        self.note_btn = IconButton(ICONS["note"], "Open the lecture note in Obsidian")
        self.note_btn.clicked.connect(self._open_note)
        self.note_btn.setVisible(False)
        foot.addWidget(self.note_btn)
        copy = IconButton(ICONS["copy"], "Copy the transcript")
        copy.clicked.connect(self._copy)
        folder = IconButton(ICONS["folder"], "Open the session folder")
        folder.clicked.connect(self._open_folder)
        self.export_btn = IconButton(ICONS["export"], "Export as text or audio")
        self.export_btn.clicked.connect(self._export_menu)
        self.settings_btn = IconButton(ICONS["settings"], "Settings")
        self.settings_btn.clicked.connect(self._toggle_sheet)
        for b in (copy, folder, self.export_btn, self.settings_btn):
            foot.addWidget(b)
        outer.addLayout(foot)

    def _build_picker(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(12)
        head = QLabel("Which module is this lecture for?")
        head.setObjectName("headline")
        head.setWordWrap(True)
        sub = QLabel("Each module has its own folder. The recording, the transcript and the exports "
                     "of every lecture are saved there.")
        sub.setObjectName("rowSupport")
        sub.setWordWrap(True)
        lay.addWidget(head)
        lay.addWidget(sub)

        holder = QWidget()
        self.module_list = QVBoxLayout(holder)
        self.module_list.setContentsMargins(0, 4, 6, 4)
        self.module_list.setSpacing(8)
        scroll = QScrollArea()
        scroll.setObjectName("plain")
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lay.addWidget(scroll, 1)

        # new-module form
        self.add_card = QFrame()
        self.add_card.setObjectName("sheet")
        al = QVBoxLayout(self.add_card)
        al.setContentsMargins(16, 14, 16, 14)
        al.setSpacing(8)
        al.addWidget(caption("New module"))
        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText("Name, e.g. Corporate Finance")
        self.new_name.textChanged.connect(self._new_name_changed)
        self.new_name.returnPressed.connect(self._add_module)
        al.addWidget(self.new_name)
        al.addWidget(caption("Folder"))
        frow = QHBoxLayout()
        self.new_folder_lbl = ElidedLabel()
        self.new_folder_lbl.setObjectName("rowSupport")
        frow.addWidget(self.new_folder_lbl, 1)
        browse = QPushButton("Choose folder…")
        browse.setObjectName("text")
        browse.clicked.connect(self._choose_folder)
        frow.addWidget(browse)
        al.addLayout(frow)
        al.addWidget(caption("Obsidian notes"))
        orow = QHBoxLayout()
        self.new_vault_lbl = ElidedLabel()
        self.new_vault_lbl.setObjectName("rowSupport")
        orow.addWidget(self.new_vault_lbl, 1)
        vbrowse = QPushButton("Choose folder…")
        vbrowse.setObjectName("text")
        vbrowse.clicked.connect(self._choose_vault_folder)
        orow.addWidget(vbrowse)
        al.addLayout(orow)
        arow = QHBoxLayout()
        arow.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setObjectName("text")
        cancel.clicked.connect(lambda: self._show_add_form(False))
        self.add_confirm = QPushButton("Add")
        self.add_confirm.setObjectName("filled")
        self.add_confirm.clicked.connect(self._add_module)
        arow.addWidget(cancel)
        arow.addWidget(self.add_confirm)
        al.addLayout(arow)
        self.add_card.setVisible(False)
        lay.addWidget(self.add_card)

        self.add_btn = QPushButton("Add module")
        self.add_btn.setIcon(glyph_icon(ICONS["add"], self.t["on_primary_container"], 18))
        self.add_btn.setObjectName("fab")
        self.add_btn.setCursor(Qt.PointingHandCursor)
        self.add_btn.clicked.connect(lambda: self._show_add_form(True))
        lay.addWidget(self.add_btn, 0, Qt.AlignLeft)
        return page

    def _style(self):
        t = self.t
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(t["surface_container"]))
        pal.setColor(QPalette.WindowText, QColor(t["on_surface"]))
        pal.setColor(QPalette.Base, QColor(t["surface_container_highest"]))
        pal.setColor(QPalette.Text, QColor(t["on_surface"]))
        pal.setColor(QPalette.Button, QColor(t["surface_container_high"]))
        pal.setColor(QPalette.ButtonText, QColor(t["on_surface"]))
        pal.setColor(QPalette.Highlight, QColor(t["secondary_container"]))
        pal.setColor(QPalette.HighlightedText, QColor(t["on_secondary_container"]))
        pal.setColor(QPalette.PlaceholderText, QColor(t["on_surface_variant"]))
        pal.setColor(QPalette.ToolTipBase, QColor(t["surface_container_highest"]))
        pal.setColor(QPalette.ToolTipText, QColor(t["on_surface"]))
        QApplication.instance().setPalette(pal)
        QApplication.instance().setFont(font(14))
        field = t["surface_container_highest"]
        self.setStyleSheet(f"""
            QLabel {{ background: transparent; }}
            QLabel#title {{ font: 600 16px "{DISPLAY_FONT}"; }}
            QLabel#headline {{ font: 500 24px "{DISPLAY_FONT}"; }}
            QScrollArea#plain, QScrollArea#plain > QWidget > QWidget {{ background: transparent; border: none; }}
            QPushButton#fab {{ background: {t['primary_container']}; color: {t['on_primary_container']}; border: none;
                               border-radius: {SHAPE['lg']}px; min-height: 56px; padding: 0 22px 0 18px;
                               font: 600 15px "{TEXT_FONT}"; }}
            QPushButton#fab:hover {{ background: {t['primary_container']}; color: {t['on_surface']}; }}
            QPushButton#filled {{ background: {t['primary']}; color: {t['on_primary']}; border: none;
                                  border-radius: 20px; min-height: 40px; padding: 0 24px; font: 600 14px "{TEXT_FONT}"; }}
            QPushButton#filled:disabled {{ background: {rgba(t['on_surface'], 0.12)}; color: {rgba(t['on_surface'], 0.38)}; }}
            QPushButton#chip {{ background: transparent; color: {t['on_surface']}; border: 1px solid {t['outline_variant']};
                                border-radius: {SHAPE['sm']}px; min-height: 32px; padding: 0 12px;
                                font: 600 14px "{TEXT_FONT}"; text-align: left; }}
            QPushButton#chip:hover {{ background: {rgba(t['on_surface'], 0.08)}; }}
            QPushButton#chip:disabled {{ color: {t['on_surface_variant']}; border-color: transparent; }}
            QMenu {{ background: {t['surface_container']}; border: 1px solid {t['outline_variant']};
                     border-radius: {SHAPE['md']}px; padding: 8px 0; }}
            QMenu::item {{ padding: 10px 20px; font-size: 14px; color: {t['on_surface']}; background: transparent; }}
            QMenu::item:selected {{ background: {rgba(t['on_surface'], 0.08)}; }}
            QMenu::item:disabled {{ color: {rgba(t['on_surface'], 0.38)}; }}
            QLabel#status {{ font-size: 14px; color: {t['on_surface_variant']}; }}
            QLabel#status[error="true"] {{ color: {t['error']}; }}
            QLabel#caption {{ font: 600 12px "{TEXT_FONT}"; color: {t['on_surface_variant']}; padding-top: 6px; }}
            QLabel#rowTitle {{ font-size: 15px; }}
            QLabel#rowSupport {{ font-size: 12px; color: {t['on_surface_variant']}; }}
            QLabel#stats {{ font-size: 12px; color: {t['on_surface_variant']}; }}
            QLabel#warn {{ font: 600 12px "{TEXT_FONT}"; color: {t['error']}; }}
            QScrollArea#sheetWrap {{ background: transparent; border: none; }}
            QScrollArea#sheetWrap > QWidget > QWidget {{ background: transparent; }}
            QFrame#sheet {{ background: {rgba(t['surface_container_high'], 0.78)}; border-radius: {SHAPE['lg_inc']}px; }}
            QTextEdit#transcript {{ background: {rgba(t['surface_container_low'], 0.4)}; border: none;
                                    border-radius: {SHAPE['lg_inc']}px; padding: 10px 12px;
                                    selection-background-color: {t['primary_container']};
                                    selection-color: {t['on_primary_container']}; }}
            QComboBox {{ background: {field}; border: none; border-radius: {SHAPE['md']}px; padding: 0 40px 0 12px;
                         min-height: 40px; font-size: 14px; }}
            QComboBox:disabled {{ color: {rgba(t['on_surface'], 0.38)}; }}
            QComboBox::drop-down {{ border: none; background: transparent; width: 28px; }}
            QComboBox QAbstractItemView {{ background: {t['surface_container']}; border: 1px solid {t['outline_variant']};
                                           padding: 4px; outline: 0; }}
            QComboBox QAbstractItemView::item {{ min-height: 36px; padding: 0 10px; border-radius: {SHAPE['sm']}px; }}
            QComboBox QAbstractItemView::item:selected {{ background: {t['secondary_container']};
                                                         color: {t['on_secondary_container']}; }}
            QLineEdit {{ background: {field}; border: 2px solid transparent; border-radius: {SHAPE['md']}px;
                         padding: 0 10px; min-height: 36px; font-size: 14px; }}
            QLineEdit:focus {{ border-color: {t['primary']}; }}
            QPushButton#tonal {{ background: {t['secondary_container']}; color: {t['on_secondary_container']};
                                 border: none; border-radius: 20px; min-height: 40px; padding: 0 20px;
                                 font: 600 14px "{TEXT_FONT}"; }}
            QPushButton#tonal:hover {{ background: {t['secondary_container']}; color: {t['on_surface']}; }}
            QPushButton#text {{ background: transparent; color: {t['primary']}; border: none; border-radius: 20px;
                                min-height: 40px; padding: 0 14px; font: 600 14px "{TEXT_FONT}"; }}
            QPushButton#text:hover {{ background: {rgba(t['primary'], 0.08)}; }}
            QPushButton#icon {{ font-family: "{ICON_FONT}"; font-size: 16px; color: {t['on_surface_variant']};
                                background: transparent; border: none; border-radius: 20px;
                                min-width: 40px; max-width: 40px; min-height: 40px; max-height: 40px; }}
            QPushButton#icon:hover {{ background: {rgba(t['on_surface'], 0.08)}; }}
            QPushButton#icon:pressed {{ background: {rgba(t['on_surface'], 0.12)}; }}
            QPushButton#icon:disabled {{ color: {rgba(t['on_surface'], 0.38)}; }}
            QPushButton#icon[selected="true"] {{ background: {t['secondary_container']};
                                                 color: {t['on_secondary_container']}; }}
            QSlider::groove:horizontal {{ height: 16px; background: {t['secondary_container']}; border-radius: 8px; }}
            QSlider::sub-page:horizontal {{ background: {t['primary']}; border-radius: 8px; }}
            QSlider::handle:horizontal {{ background: {t['primary']}; width: 4px; margin: -12px 4px;
                                          border-radius: 2px; }}
            QScrollBar:vertical {{ background: transparent; width: 10px; margin: 12px 2px; }}
            QScrollBar::handle:vertical {{ background: {t['outline_variant']}; border-radius: 3px; min-height: 40px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
            QToolTip {{ background: {t['surface_container_highest']}; color: {t['on_surface']}; border: none;
                        padding: 6px 8px; border-radius: {SHAPE['xs']}px; }}
        """)
        self.ts_fmt = QTextCharFormat()
        self.ts_fmt.setForeground(QColor(t["outline"]))
        self.ts_fmt.setFont(font(12, QFont.Weight.DemiBold, tabular=True))
        self.final_fmt = QTextCharFormat()
        self.final_fmt.setForeground(QColor(t["on_surface"]))
        self.final_fmt.setFont(font(16))
        self.partial_fmt = QTextCharFormat(self.final_fmt)
        self.partial_fmt.setForeground(QColor(t["on_surface_variant"]))
        self.partial_fmt.setFontItalic(True)
        self.block_fmt = QTextBlockFormat()
        self.block_fmt.setBottomMargin(12)
        self.block_fmt.setLineHeight(140, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)

    def paintEvent(self, _):
        """Translucent surface, rounded on the left (extra-large-increased), flush with the screen edge."""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = SHAPE["xl_inc"]
        path = QPainterPath()
        path.addRoundedRect(QRectF(0.5, 0.5, self.width() + r, self.height() - 1), r, r)
        p.fillPath(path, qcolor(self.t["surface"], self.cfg.background_opacity))
        p.setPen(QPen(qcolor(self.t["outline_variant"], 0.8), 1))
        p.drawPath(path)
        p.end()

    @staticmethod
    def _set_prop(w: QWidget, name: str, value: bool):
        w.setProperty(name, "true" if value else "false")
        w.style().unpolish(w)
        w.style().polish(w)

    # ------------------------------------------------------------------ docking
    def _dock(self):
        try:
            if self.dock is None:
                self.dock = AppBar(int(self.winId()), self.cfg.panel_width)
                atexit.register(self.dock.remove)
            self.dock.register()
        except Exception as exc:
            self._error_box("Could not dock to the screen edge", str(exc), traceback.format_exc())

    def nativeEvent(self, event_type, message):
        if self.dock is not None and bytes(event_type) == b"windows_generic_MSG":
            try:
                if self.dock.handle(int(message)):
                    return True, 0
            except Exception as exc:
                self._error_box("Could not dock to the screen edge", str(exc), traceback.format_exc())
        return super().nativeEvent(event_type, message)

    def changeEvent(self, ev):
        if ev.type() == QEvent.Type.WindowStateChange and self.dock is not None:
            if self.windowState() & Qt.WindowMinimized:
                self.dock.remove()          # give the strip back while minimized
            elif not self.dock.registered:
                QTimer.singleShot(0, self._dock)
        super().changeEvent(ev)

    def _toggle_sheet(self):
        self.sheet_open = not self.sheet_open
        self._set_prop(self.settings_btn, "selected", self.sheet_open)
        self.sheet_anim.stop()
        start = self.sheet_wrap.height() if self.sheet_wrap.isVisible() else 0
        if self.sheet_open:
            self.sheet_wrap.setVisible(True)
            end = min(self.sheet.sizeHint().height(), round(self.height() * 0.62))
        else:
            end = 0
        self.sheet_anim.setStartValue(start)
        self.sheet_anim.setEndValue(end)
        self.sheet_anim.start()

    def _sheet_anim_done(self):
        if not self.sheet_open:
            self.sheet_wrap.setVisible(False)

    # ------------------------------------------------------------------ modules
    def _show_picker(self):
        if self.state != "idle":
            return
        self._fill_modules()
        self.module_chip.setVisible(False)
        self._show_add_form(not self.store.modules)
        self.stack.setCurrentIndex(0)

    def _fill_modules(self):
        while self.module_list.count():
            w = self.module_list.takeAt(0).widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        for m in self.store.modules:
            item = ModuleItem(self.t, m, current=m.name == self.store.last, vault=self.cfg.vault_path)
            item.clicked.connect(lambda _=False, name=m.name: self._pick_module(name))
            item.remove_requested.connect(self._remove_module)
            self.module_list.addWidget(item)
        if not self.store.modules:
            empty = QLabel("No modules yet. Add the courses you record, one per module.")
            empty.setObjectName("rowSupport")
            empty.setWordWrap(True)
            self.module_list.addWidget(empty)
        self.module_list.addStretch(1)

    def _pick_module(self, name: str):
        try:
            self.module = self.store.use(name)
        except Exception as exc:
            self._error_box("Could not open the module", str(exc), traceback.format_exc())
            return
        self.module_chip.setText(f"{self.module.name}")
        self.module_chip.setIcon(glyph_icon(ICONS["folder"], self.t["on_surface_variant"], 14))
        self.module_chip.setToolTip(f"{self.module.folder}\nClick to change module")
        self.module_chip.setVisible(True)
        self.stack.setCurrentIndex(1)
        self._set_state(self.state)

    def _show_add_form(self, show: bool):
        self.add_card.setVisible(show)
        self.add_btn.setVisible(not show)
        if show:
            self.new_name.clear()
            self.new_folder = None
            self.new_vault_folder = None
            self._new_name_changed("")
            self.new_name.setFocus()

    def _default_vault_folder(self, name: str) -> Path | None:
        vault = self.cfg.vault_path
        return vault / self.cfg.obsidian_modules / name if vault is not None else None

    def _new_name_changed(self, text: str):
        name = text.strip() or "…"
        if self.new_folder is None:
            self.new_folder_lbl.set_full_text(str(self.store.default_root / name))
        if self.new_vault_folder is None:
            v = self._default_vault_folder(name)
            self.new_vault_lbl.set_full_text(str(v) if v else "No vault set (obsidian_vault in config.yaml)")
        self.add_confirm.setEnabled(bool(text.strip()))

    def _choose_vault_folder(self):
        start = str(self.new_vault_folder or self.cfg.vault_path or Path.home())
        picked = QFileDialog.getExistingDirectory(self, "Obsidian folder for this module's notes", start)
        if picked:
            self.new_vault_folder = Path(picked)
            self.new_vault_lbl.set_full_text(picked)

    def _choose_folder(self):
        start = str(self.new_folder or self.store.default_root)
        picked = QFileDialog.getExistingDirectory(self, "Folder for this module", start)
        if picked:
            self.new_folder = Path(picked)
            self.new_folder_lbl.set_full_text(picked)

    def _add_module(self):
        try:
            name = check_name(self.new_name.text())
            vault_folder = self.new_vault_folder or self._default_vault_folder(name)
            module = self.store.add(name, self.new_folder, vault_folder)
        except ValueError as exc:
            self._error_box("Can't add this module", str(exc))
            return
        except Exception as exc:
            self._error_box("Can't add this module", str(exc), traceback.format_exc())
            return
        self._show_add_form(False)
        self._pick_module(module.name)

    def _remove_module(self, name: str):
        answer = QMessageBox.question(self, APP, f"Remove “{name}” from the list?\n\n"
                                      "Its folder and all recordings in it stay on disk.")
        if answer == QMessageBox.Yes:
            self.store.remove(name)
            self._fill_modules()

    # ------------------------------------------------------------------ export
    def _export_menu(self):
        menu = QMenu(self)
        menu.setWindowFlags(menu.windowFlags() | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        menu.setAttribute(Qt.WA_TranslucentBackground)
        for kind, (label, _filt, _ext, _fn) in export.KINDS.items():
            menu.addAction(label, lambda k=kind: self._export(k))
        menu.addSeparator()
        act = menu.addAction("Lecture note in Obsidian (Qwen)", self._note_for_session)
        act.setEnabled(not self.note_busy and self.cfg.vault_path is not None)
        size = menu.sizeHint()
        corner = self.export_btn.mapToGlobal(self.export_btn.rect().topRight())
        menu.exec(QPoint(corner.x() - size.width(), corner.y() - size.height() - 6))

    def _export(self, kind: str):
        label, filt, ext, fn = export.KINDS[kind]
        folder = self.session_folder
        if folder is None or not (folder / "transcript.txt").is_file():
            start = str(self.module.path if self.module else self.cfg.output_path)
            picked = QFileDialog.getExistingDirectory(self, "Choose a recorded lecture to export", start)
            if not picked:
                return
            folder = Path(picked)
        if kind == "recording" and self.state != "idle" and folder == self.session_folder:
            self._error_box("Still recording", "Stop the recording before exporting its audio.")
            return
        prefix = f"{self.module.name} " if self.module else ""
        dest, _ = QFileDialog.getSaveFileName(self, f"Export: {label}", str(folder / f"{prefix}{folder.name}{ext}"), filt)
        if not dest:
            return
        self.stats_lbl.setText(f"Exporting {label.lower()}…")
        self.export_btn.setEnabled(False)

        def work():
            try:
                fn(folder, Path(dest))
                self.bridge.event.emit("export_done", Path(dest))
            except Exception as exc:
                self.bridge.event.emit("export_error", (f"{type(exc).__name__}: {exc}", traceback.format_exc()))

        threading.Thread(target=work, name="export", daemon=True).start()

    # ------------------------------------------------------------------ obsidian notes
    def _note_for_session(self):
        folder = self.session_folder
        if folder is None or self.state != "idle":
            start = str(self.module.path if self.module else self.cfg.output_path)
            picked = QFileDialog.getExistingDirectory(self, "Choose a recorded lecture", start)
            if not picked:
                return
            folder = Path(picked)
        module = next((m for m in self.store.modules if folder.parent == m.path), None)
        if module is None:
            self._error_box("No module for this lecture",
                            f"{folder} is not inside a module's folder, so it is unclear which Obsidian folder "
                            "the note belongs in.")
            return
        self._make_note(folder, module)

    def _make_note(self, folder: Path, module: Module):
        if module.vault_path is None:
            self._error_box("No Obsidian folder", f"The module “{module.name}” has no Obsidian folder.")
            return
        self.note_busy = True
        self.note_btn.setVisible(False)
        self._status("Writing the Obsidian note…")

        def work():
            try:
                path = notes.write_note(self.cfg, module, folder,
                                        lambda text: self.bridge.event.emit("note_progress", text))
                self.bridge.event.emit("note_done", path)
            except Exception as exc:
                self.bridge.event.emit("note_error", (f"{type(exc).__name__}: {exc}", traceback.format_exc()))

        threading.Thread(target=work, name="note", daemon=True).start()

    def _open_note(self):
        if self.note_path is None:
            return
        from urllib.parse import quote
        if not QDesktopServices.openUrl(QUrl("obsidian://open?path=" + quote(str(self.note_path), safe=""))):
            self._error_box("Could not open Obsidian",
                            f"Windows has no app for obsidian:// links. The note is here:\n{self.note_path}")

    def _note_finished(self):
        self.note_busy = False
        if self.quit_after_finish and self.state == "idle":
            self.close()

    # ------------------------------------------------------------------ startup
    def _startup(self):
        self._show_picker()
        self._dock()
        self._refresh_devices(initial=True)
        need = downloads.absent(self.cfg)
        if need:
            self._offer_download(need)
        else:
            threading.Thread(target=self._load_engine, name="load", daemon=True).start()

    def _offer_download(self, need: list):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            sizes = [downloads.remote_size(f.url) for f in need]
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            self._status("Models missing", error=True)
            self._error_box("Models missing", "Lecture Live needs to download its models once, but the download "
                            f"server can't be reached:\n{exc}\n\nConnect to the internet and start the app again.",
                            traceback.format_exc())
            return
        QApplication.restoreOverrideCursor()
        listing = "\n".join(f"•  {f.what}: {s / 1e9:.2f} GB" for f, s in zip(need, sizes))
        answer = QMessageBox.question(
            self, APP, f"Lecture Live needs to download these models once ({sum(sizes) / 1e9:.1f} GB):\n\n"
            f"{listing}\n\nThey are kept in {self.cfg.models_path}. Download now?\n"
            "An interrupted download continues where it stopped next time.")
        if answer != QMessageBox.Yes:
            self._status("Models missing; start the app again to download them", error=True)
            return

        def work():
            try:
                for f, total in zip(need, sizes):
                    shown = [-1]

                    def progress(have, f=f, total=total):
                        pct = int(have * 100 / total)
                        if pct != shown[0]:
                            shown[0] = pct
                            self.bridge.event.emit("dl_progress", f"Downloading {f.what} · {pct}% "
                                                                  f"({have / 1e9:.2f} of {total / 1e9:.2f} GB)")
                    downloads.download(f, total, progress)
                self.bridge.event.emit("dl_done", None)
            except Exception as exc:
                self.bridge.event.emit("dl_error", (f"{type(exc).__name__}: {exc}", traceback.format_exc()))

        threading.Thread(target=work, name="download", daemon=True).start()

    def _load_engine(self):
        try:
            self.live.load_engine()
            self.bridge.event.emit("engine_ready", None)
        except Exception as exc:
            self.bridge.event.emit("engine_error", (f"{type(exc).__name__}: {exc}", traceback.format_exc()))

    def _fill_languages(self):
        codes = language_codes()
        self.lang_box.addItem("Auto-detect", "")
        for c in sorted((c for c in codes if c in LANGUAGE_NAMES), key=lambda c: LANGUAGE_NAMES[c]):
            self.lang_box.addItem(f"{LANGUAGE_NAMES[c]} ({c})", c)
        self.lang_box.insertSeparator(self.lang_box.count())
        for c in codes:
            if c not in LANGUAGE_NAMES:
                self.lang_box.addItem(c, c)
        idx = self.lang_box.findData(self.cfg.lang)
        if idx < 0:
            raise ValueError(f"config.yaml: lang {self.cfg.lang!r} is not a Whisper language code")
        self.lang_box.setCurrentIndex(idx)

    def _refresh_devices(self, initial: bool = False):
        if self.state != "idle":
            return
        current = self.live.capture.device.name if self.live.capture else (self.cfg.input_device or None)
        try:
            if not initial:
                self.live.close_input()
                rescan()
            self.devices, default = list_input_devices()
        except Exception as exc:
            self._error_box("No microphone", str(exc), traceback.format_exc())
            return
        self.dev_box.blockSignals(True)
        self.dev_box.clear()
        for d in self.devices:
            self.dev_box.addItem(d.name, d.index)
            self.dev_box.setItemData(self.dev_box.count() - 1, f"{d.channels} ch, {d.samplerate} Hz", Qt.ToolTipRole)
        self.dev_box.blockSignals(False)
        if current is None:
            pick = next((i for i, d in enumerate(self.devices) if d.index == default), 0)
        else:
            pick = next((i for i, d in enumerate(self.devices) if d.name == current), None)
            if pick is None:
                self.dev_box.setCurrentIndex(-1)
                self._error_box("Microphone not found",
                                f"The microphone “{current}” is not connected.\nChoose one in Settings.")
                return
        self.dev_box.setCurrentIndex(pick)
        if initial or self.live.capture is None or self.live.capture.device != self.devices[pick]:
            self._open_device(self.devices[pick])

    def _open_device(self, dev: InputDevice):
        try:
            self.live.open_input(dev, self.raw_sw.isChecked())
            self.cfg.input_device = dev.name
            self.cfg.raw_capture = self.raw_sw.isChecked()
        except Exception as exc:
            self.dev_box.setCurrentIndex(-1)
            hint = ("\n\nRaw microphone is on; this driver may not support it. Switch it off in Settings "
                    "and choose the microphone again." if self.raw_sw.isChecked() else "")
            self._error_box("Could not open the microphone", f"{dev.name}: {exc}{hint}", traceback.format_exc())

    # ------------------------------------------------------------------ control handlers
    def _device_chosen(self, i: int):
        if 0 <= i < len(self.devices):
            self._open_device(self.devices[i])

    def _raw_toggled(self, _):
        i = self.dev_box.currentIndex()
        if 0 <= i < len(self.devices):
            self._open_device(self.devices[i])

    def _mode_changed(self, value: str):
        self.cfg.segmentation = value

    def _profile_changed(self, value: str):
        self.cfg.compute_profile = value
        self.live.profile = PROFILES[value]

    def _lang_changed(self, _):
        self.live.language = self.lang_box.currentData()
        self.cfg.lang = self.live.language

    def _gain_changed(self, v: int):
        self.live.gain_db = float(v)
        self.gain_lbl.setText(f"{v:+d} dB")

    def _hear(self):
        try:
            self.live.play_last(5.0)
        except Exception as exc:
            self._error_box("Playback failed", str(exc), traceback.format_exc())

    def _copy(self):
        QGuiApplication.clipboard().setText("\n".join(self.final_lines))

    def _open_folder(self):
        folder = self.session_folder or (self.module.path if self.module else self.cfg.output_path)
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _show_log(self):
        if self.log_dialog is None:
            self.log_dialog = LogDialog(self, self.log_lines)
        self.log_dialog.show()
        self.log_dialog.raise_()

    def _toggle(self):
        if self.state == "idle" and self.rec_btn.isEnabled():
            self._start()
        elif self.state == "recording":
            self._stop()

    def _start(self):
        try:
            self._save_settings()
            self.session_folder = self.live.start_session(self.module)
        except Exception as exc:
            self._error_box("Could not start", str(exc), traceback.format_exc())
            return
        self.text.clear()
        self.final_lines.clear()
        self.partial_pos = None
        self.next_final_id = 0
        self.last_speaker = None
        self.overflows = 0
        self.warn_lbl.setVisible(False)
        self.started_at = time.monotonic()
        self._set_state("recording")

    def _stop(self):
        self._set_state("stopping")
        self.live.stop_session()

    def _set_state(self, state: str):
        self.state = state
        self.rec_btn.setEnabled(state != "stopping" and self.engine_ok and self.module is not None)
        self.module_chip.setEnabled(state == "idle")
        self.rec_btn.set_mode(state)
        if state == "recording":
            self._status("Listening")
        elif state == "stopping":
            self._status("Finishing…")
        elif self.engine_ok:
            self._status("Ready")
        locked = state != "idle"
        for w in (self.dev_box, self.refresh_btn, self.raw_sw, self.mode):
            w.setEnabled(not locked)

    def _status(self, text: str, error: bool = False):
        self.status_lbl.setText(text)
        self._set_prop(self.status_lbl, "error", error)

    def _tick(self):
        if self.state == "recording":
            self.clock_lbl.setText(hms(time.monotonic() - self.started_at))

    # ------------------------------------------------------------------ events from Live
    def _on_event(self, kind: str, payload):
        if kind == "level":
            self.meter.set_level(payload)
            self.rec_btn.set_level(payload.rms_db)
        elif kind == "partial":
            self._on_partial(payload)
        elif kind == "final":
            self._on_final(payload)
        elif kind == "status":
            self._on_status(payload)
        elif kind == "log":
            self.log_lines.append(payload)
            if self.log_dialog is not None:
                self.log_dialog.append(payload)
        elif kind == "warning":
            self.overflows += 1
            self.warn_lbl.setText(payload + (f" ({self.overflows}×)" if self.overflows > 1 else ""))
            self.warn_lbl.setVisible(True)
        elif kind == "error":
            self._on_error(*payload)
        elif kind == "finished":
            self._on_finished(payload)
        elif kind == "engine_ready":
            self.engine_ok = True
            e = self.live.engine
            self.stats_lbl.setText(f"{e.model_name} · {e.gpu.split(':')[0]}")
            self.stats_lbl.setToolTip(e.gpu)
            if self.live.tracker is None:
                self.spk_sw.setChecked(False)
                self.spk_sw.setEnabled(False)
                self.spk_sw.setToolTip(f"Speaker model not found: {self.cfg.speaker_model_path}")
            self._set_state(self.state)
        elif kind == "dl_progress":
            self._status(payload)
        elif kind == "dl_done":
            self._status("Loading Whisper…")
            threading.Thread(target=self._load_engine, name="load", daemon=True).start()
        elif kind == "dl_error":
            message, tb = payload
            self._status("Model download failed", error=True)
            self._error_box("Model download failed", message + "\n\nStart the app again to continue the download.", tb)
        elif kind == "note_progress":
            if self.state == "idle":
                self._status(payload)
        elif kind == "note_done":
            self.note_path = payload
            self.note_btn.setVisible(True)
            self.note_btn.setToolTip(f"Open in Obsidian: {payload.name}")
            if self.state == "idle":
                self._status(f"In Obsidian · {payload.stem}")
            self._note_finished()
        elif kind == "note_error":
            message, tb = payload
            if self.state == "idle":
                self._status("The Obsidian note failed", error=True)
            self._note_finished()
            self._error_box("Obsidian note failed", message + "\n\nThe recording and transcript are saved.", tb)
        elif kind == "export_done":
            self.export_btn.setEnabled(True)
            self.stats_lbl.setText(f"Exported · {payload.name}")
            self.stats_lbl.setToolTip(str(payload))
        elif kind == "export_error":
            message, tb = payload
            self.export_btn.setEnabled(True)
            self.stats_lbl.setText("Export failed")
            self._error_box("Export failed", message, tb)
        elif kind == "engine_error":
            message, tb = payload
            self._status("Whisper is not loaded", error=True)
            self._error_box("Whisper could not start", message, tb)

    def _at_bottom(self) -> bool:
        sb = self.text.verticalScrollBar()
        return sb.value() >= sb.maximum() - 4

    def _scroll(self, was_at_bottom: bool):
        if was_at_bottom:
            sb = self.text.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _remove_partial(self):
        if self.partial_pos is None:
            return
        c = QTextCursor(self.text.document())
        c.setPosition(self.partial_pos)
        c.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        c.removeSelectedText()
        self.partial_pos = None

    def _append_block(self) -> QTextCursor:
        c = QTextCursor(self.text.document())
        c.movePosition(QTextCursor.End)
        if not self.text.document().isEmpty():
            c.insertBlock(self.block_fmt)
        else:
            c.setBlockFormat(self.block_fmt)
        return c

    def _on_partial(self, p: Partial):
        if self.state == "idle" or p.chunk_id < self.next_final_id:
            return  # the final text for that audio is already shown
        bottom = self._at_bottom()
        self._remove_partial()
        c = QTextCursor(self.text.document())
        c.movePosition(QTextCursor.End)
        self.partial_pos = c.position()
        c = self._append_block()
        c.insertText(p.text, self.partial_fmt)
        self._scroll(bottom)

    def _on_final(self, f: Final):
        bottom = self._at_bottom()
        self._remove_partial()
        for run in f.runs:
            c = self._append_block()
            if run.speaker is not None and run.speaker != self.last_speaker:
                fmt = QTextCharFormat(self.ts_fmt)
                fmt.setForeground(QColor(self.t[SPEAKER_COLORS[(run.speaker - 1) % len(SPEAKER_COLORS)]]))
                fmt.setFont(font(13, QFont.Weight.Bold))
                c.insertText(f"Speaker {run.speaker}  ", fmt)
                self.last_speaker = run.speaker
            sep = "" if run.text[:1].isspace() else " "
            c.insertText(f"[{hms(run.start)}]{sep}", self.ts_fmt)
            c.insertText(run.text, self.final_fmt)
            self.final_lines.append(line(run))
        self.next_final_id = f.chunk_id + 1
        self._scroll(bottom)

    def _on_status(self, s: Status):
        parts = []
        if s.last_rtf is not None:
            parts.append(f"{s.last_rtf:.2f}× real time")
        if s.backlog_s >= 1:
            parts.append(f"{s.backlog_s:.0f} s behind")
        parts.append(f"{s.chunks} chunks")
        self.stats_lbl.setText(" · ".join(parts))
        if self.state == "stopping":
            self._status(f"Finishing · {s.backlog_s:.0f} s left")

    def _on_error(self, where: str, tb: str, folder: Path | None):
        self._remove_partial()
        saved = f"\n\nEverything transcribed before the error is saved in:\n{folder}" if folder else ""
        if where == "audio":
            self.dev_box.setCurrentIndex(-1)
            saved += "\n\nThe microphone was closed. Choose it again in Settings to reopen it."
        self._set_state("idle")
        self._status(f"Stopped by an error in {where}", error=True)
        self._error_box(f"Error in {where}", tb.strip().splitlines()[-1] + saved, tb)
        if self.quit_after_finish:
            self.close()

    def _on_finished(self, folder: Path):
        self._remove_partial()
        self._set_state("idle")
        self._status(f"Saved · {folder.name}")
        self.status_lbl.setToolTip(str(folder))
        # straight into Obsidian when the module has a vault folder
        if self.module is not None and self.module.vault_path is not None and self.cfg.vault_path is not None:
            self._make_note(folder, self.module)
        if self.quit_after_finish:
            self.close()

    def _error_box(self, title: str, text: str, details: str | None = None):
        box = QMessageBox(QMessageBox.Critical, title, text, QMessageBox.Ok, self)
        if details:
            box.setDetailedText(details)
        box.exec()

    # ------------------------------------------------------------------ closing
    def _save_settings(self):
        self.cfg.input_gain_db = float(self.gain.value())
        self.cfg.agc = self.agc_sw.isChecked()
        self.cfg.initial_prompt = self.prompt.text()
        self.cfg.use_context = self.ctx_sw.isChecked()
        if self.spk_sw.isEnabled():
            self.cfg.speakers = self.spk_sw.isChecked()
        config.save({k: getattr(self.cfg, k) for k in (
            "input_device", "raw_capture", "input_gain_db", "agc", "lang", "segmentation",
            "initial_prompt", "use_context", "speakers", "compute_profile")})

    def closeEvent(self, ev):
        if self.note_busy:
            self.quit_after_finish = True
            self._status("Finishing the Obsidian note, then closing…")
            ev.ignore()
            return
        if self.state != "idle":
            if self.state == "recording":
                answer = QMessageBox.question(self, APP, "A lecture is being recorded.\n\n"
                                              "Stop, finish transcribing what was recorded, and quit?")
                if answer != QMessageBox.Yes:
                    ev.ignore()
                    return
                self._stop()
            self.quit_after_finish = True
            ev.ignore()
            return
        try:
            self._save_settings()
        finally:
            if self.dock is not None:
                self.dock.remove()
            self.live.shutdown()
        ev.accept()


def main():
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(make_icon()))
    try:
        cfg = config.load()
        win = MainWindow(cfg, pick_scheme())
    except Exception as exc:
        box = QMessageBox(QMessageBox.Critical, APP, f"{APP} could not start:\n\n{exc}", QMessageBox.Ok)
        box.setDetailedText(traceback.format_exc())
        box.exec()
        return 1
    win.show()
    return app.exec()
