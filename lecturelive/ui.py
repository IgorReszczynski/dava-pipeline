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
                           QPainterPath, QPalette, QPen, QPixmap, QShortcut, QTextBlockFormat, QTextCharFormat,
                           QTextCursor)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QButtonGroup, QComboBox, QDialog, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QListView, QMenu, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QSizePolicy, QSlider, QStackedWidget, QTextEdit, QVBoxLayout,
                               QWidget)

from . import config, export
from .audio import InputDevice, list_input_devices, rescan
from .dock import AppBar
from .engine import language_codes
from .live import Final, Level, Live, Partial, Status, hms, line
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
             refresh="", export="", remove="", chevron="", add="")
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
class RecordButton(QAbstractButton):
    """Circle when idle; morphs to a rounded square (large-increased, 20) while recording."""
    SIZE = 72

    def __init__(self, t: dict):
        super().__init__()
        self.t = t
        self.mode = "idle"          # idle | recording | stopping
        self._radius = self.SIZE / 2
        self._spin = 0.0
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover)
        self._morph = QVariantAnimation(self, duration=450, easingCurve=QEasingCurve.OutBack)
        self._morph.valueChanged.connect(self._set_radius)
        self._spinner = QTimer(self, interval=16, timeout=self._advance)
        self.pressed.connect(lambda: self._animate_to(self._target() - 6))
        self.released.connect(lambda: self._animate_to(self._target()))

    def _target(self) -> float:
        return self.SIZE / 2 if self.mode == "idle" else SHAPE["lg_inc"]

    def _set_radius(self, r):
        self._radius = float(r)
        self.update()

    def _animate_to(self, r: float):
        self._morph.stop()
        self._morph.setStartValue(self._radius)
        self._morph.setEndValue(float(r))
        self._morph.start()

    def _advance(self):
        self._spin = (self._spin + 6) % 360
        self.update()

    def set_mode(self, mode: str):
        self.mode = mode
        self._animate_to(self._target())
        if mode == "stopping":
            self._spinner.start()
        else:
            self._spinner.stop()

    def paintEvent(self, _):
        t = self.t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect())
        if not self.isEnabled():
            bg, fg = qcolor(t["on_surface"], 0.12), qcolor(t["on_surface"], 0.38)
        elif self.mode == "recording":
            bg, fg = QColor(t["error"]), QColor(t["on_error"])
        elif self.mode == "stopping":
            bg, fg = QColor(t["surface_container_highest"]), QColor(t["primary"])
        else:
            bg, fg = QColor(t["primary"]), QColor(t["on_primary"])
        r = min(self._radius, rect.height() / 2)
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(rect, r, r)
        if self.isEnabled() and (self.underMouse() or self.isDown()):
            p.setBrush(qcolor(fg.name(), 0.12 if self.isDown() else 0.08))
            p.drawRoundedRect(rect, r, r)
        c = rect.center()
        p.setBrush(fg)
        if self.mode == "recording":
            p.drawRoundedRect(QRectF(c.x() - 12, c.y() - 12, 24, 24), 6, 6)
        elif self.mode == "stopping":
            pen = QPen(fg)
            pen.setWidthF(4)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawArc(QRectF(c.x() - 14, c.y() - 14, 28, 28), int(-self._spin * 16), 270 * 16)
        else:
            p.drawEllipse(c, 13, 13)
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

    def __init__(self, t: dict, module: Module, current: bool):
        super().__init__()
        self.t = t
        self.module = module
        self.current = current
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover)
        self.setMinimumHeight(72)
        self.setToolTip(f"Record into {module.folder}")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(68, 10, 8, 10)
        texts = QVBoxLayout()
        texts.setSpacing(0)
        name = QLabel(module.name)
        name.setObjectName("rowTitle")
        folder = ElidedLabel(module.folder)
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
        self.title_lbl = QLabel(APP)
        self.title_lbl.setObjectName("title")
        top.addWidget(self.title_lbl)
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
            QFrame#sheet {{ background: {rgba(t['surface_container_high'], 0.92)}; border-radius: {SHAPE['lg_inc']}px; }}
            QTextEdit#transcript {{ background: {rgba(t['surface_container_low'], 0.55)}; border: none;
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
        self.title_lbl.setVisible(True)
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
            item = ModuleItem(self.t, m, current=m.name == self.store.last)
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
        self.title_lbl.setVisible(False)
        self.module_chip.setVisible(True)
        self.stack.setCurrentIndex(1)
        self._set_state(self.state)

    def _show_add_form(self, show: bool):
        self.add_card.setVisible(show)
        self.add_btn.setVisible(not show)
        if show:
            self.new_name.clear()
            self.new_folder = None
            self._new_name_changed("")
            self.new_name.setFocus()

    def _new_name_changed(self, text: str):
        if self.new_folder is None:
            name = text.strip() or "…"
            self.new_folder_lbl.set_full_text(str(self.store.default_root / name))
        self.add_confirm.setEnabled(bool(text.strip()))

    def _choose_folder(self):
        start = str(self.new_folder or self.store.default_root)
        picked = QFileDialog.getExistingDirectory(self, "Folder for this module", start)
        if picked:
            self.new_folder = Path(picked)
            self.new_folder_lbl.set_full_text(picked)

    def _add_module(self):
        try:
            check_name(self.new_name.text())
            module = self.store.add(self.new_name.text(), self.new_folder)
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

    # ------------------------------------------------------------------ startup
    def _startup(self):
        self._show_picker()
        self._dock()
        self._refresh_devices(initial=True)
        threading.Thread(target=self._load_engine, name="load", daemon=True).start()

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
            "initial_prompt", "use_context", "speakers")})

    def closeEvent(self, ev):
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
