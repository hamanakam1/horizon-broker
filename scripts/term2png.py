#!/usr/bin/env python3
"""Shared terminal-screenshot renderer.

This is the canonical implementation used by the per-repo
``scripts/make_screenshots.py`` generators. Every pixel in a generated PNG
comes from a real CLI invocation: the script drives the tool, captures the
bytes it writes to stdout/stderr, and paints those bytes into a window frame.

Nothing here is hand-drawn and nothing is pasted from a terminal that
happened to look good.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - developer tooling
    raise SystemExit(
        "Pillow is required to render screenshots: python3 -m pip install pillow"
    )

# --------------------------------------------------------------------------
# palette
# --------------------------------------------------------------------------
BACKGROUND = (13, 17, 23)
TITLEBAR = (22, 27, 34)
BORDER = (48, 54, 61)
FOREGROUND = (201, 209, 217)
DIM = (110, 118, 129)
PROMPT = (126, 231, 135)

ANSI_COLORS = {
    30: (72, 79, 88),
    31: (248, 81, 73),
    32: (63, 185, 80),
    33: (210, 153, 34),
    34: (88, 166, 255),
    35: (188, 140, 255),
    36: (57, 197, 207),
    37: (201, 209, 217),
    90: (110, 118, 129),
    91: (255, 123, 114),
    92: (86, 211, 100),
    93: (233, 196, 106),
    94: (121, 192, 255),
    95: (210, 168, 255),
    96: (118, 224, 232),
    97: (240, 246, 252),
}

FONT_SIZE = 15
LINE_SPACING = 1.45
WINDOW_PADDING = 18
TITLEBAR_HEIGHT = 30

MONO_FONTS = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/DejaVuSansMono.ttf",
)
FALLBACK_FONTS = (
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
)

SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")
OTHER_ESC_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-ln-z]")


@dataclass(frozen=True)
class Cell:
    """One character cell: what to draw, and in what style."""

    ch: str
    fg: tuple[int, int, int] | None = None
    bg: tuple[int, int, int] | None = None
    bold: bool = False
    dim: bool = False


@dataclass(frozen=True)
class Style:
    fg: tuple[int, int, int] | None = None
    bg: tuple[int, int, int] | None = None
    bold: bool = False
    dim: bool = False


def _xterm256(index: int) -> tuple[int, int, int]:
    if index < 16:
        base = [
            (72, 79, 88), (248, 81, 73), (63, 185, 80), (210, 153, 34),
            (88, 166, 255), (188, 140, 255), (57, 197, 207), (201, 209, 217),
            (110, 118, 129), (255, 123, 114), (86, 211, 100), (245, 245, 67),
            (59, 142, 234), (214, 112, 214), (41, 184, 219), (255, 255, 255),
        ]
        return base[index]
    if index < 232:
        index -= 16
        steps = [0, 95, 135, 175, 215, 255]
        return (
            steps[index // 36],
            steps[(index // 6) % 6],
            steps[index % 6],
        )
    grey = 8 + (index - 232) * 10
    return (grey, grey, grey)


def _apply_sgr(style: Style, codes: list[int]) -> Style:
    index = 0
    while index < len(codes):
        code = codes[index]
        if code == 0:
            style = Style()
        elif code == 1:
            style = replace(style, bold=True)
        elif code == 2:
            style = replace(style, dim=True)
        elif code == 22:
            style = replace(style, bold=False, dim=False)
        elif code == 39:
            style = replace(style, fg=None)
        elif code == 49:
            style = replace(style, bg=None)
        elif code in ANSI_COLORS:
            style = replace(style, fg=ANSI_COLORS[code])
        elif 40 <= code <= 47:
            style = replace(style, bg=ANSI_COLORS[code - 10])
        elif 100 <= code <= 107:
            style = replace(style, bg=ANSI_COLORS[code - 10 + 60])
        elif code in (38, 48):
            if index + 2 < len(codes) and codes[index + 1] == 5:
                colour = _xterm256(codes[index + 2])
                index += 2
            elif index + 4 < len(codes) and codes[index + 1] == 2:
                colour = (codes[index + 2], codes[index + 3], codes[index + 4])
                index += 4
            else:
                index += 1
                continue
            style = replace(style, fg=colour if code == 38 else style.fg)
            style = replace(style, bg=colour if code == 48 else style.bg)
        index += 1
    return style


def _emit(lines: list[list[Cell]], chunk: str, style: Style) -> None:
    for index, raw_line in enumerate(chunk.split("\n")):
        if index:
            lines.append([])
        if raw_line:
            lines[-1].extend(
                Cell(ch, style.fg, style.bg, style.bold, style.dim) for ch in raw_line
            )


def parse_ansi(text: str) -> list[list[Cell]]:
    """Turn ANSI-coloured text into a grid of styled cells."""

    text = OTHER_ESC_RE.sub("", text)
    lines: list[list[Cell]] = [[]]
    style = Style()
    position = 0
    for match in SGR_RE.finditer(text):
        _emit(lines, text[position : match.start()], style)
        raw = match.group(1)
        parts = raw.split(";") if raw else ["0"]
        style = _apply_sgr(style, [int(part) if part else 0 for part in parts])
        position = match.end()
    _emit(lines, text[position:], style)
    while lines and not lines[-1]:
        lines.pop()
    return lines


class FontBook:
    """Menlo plus a fallback chain, because one glyph is in nobody's font."""

    def __init__(self) -> None:
        self.regular: ImageFont.FreeTypeFont | None = None
        self.bold: ImageFont.FreeTypeFont | None = None
        self.title: ImageFont.FreeTypeFont | None = None
        for candidate in MONO_FONTS:
            if Path(candidate).exists():
                self.regular = ImageFont.truetype(candidate, FONT_SIZE, index=0)
                try:
                    self.bold = ImageFont.truetype(candidate, FONT_SIZE, index=1)
                except (OSError, ValueError):
                    self.bold = self.regular
                try:
                    self.title = ImageFont.truetype(candidate, 12, index=0)
                except (OSError, ValueError):
                    self.title = self.regular
                break
        if self.regular is None:  # pragma: no cover - depends on the host
            raise SystemExit("no monospace font found; install DejaVu Sans Mono")
        self.fallbacks: list[ImageFont.FreeTypeFont] = []
        for candidate in FALLBACK_FONTS:
            if Path(candidate).exists():
                self.fallbacks.append(ImageFont.truetype(candidate, FONT_SIZE, index=0))
        self._missing = self._notdef_signature(self.regular)
        self._cache: dict[tuple[str, bool], ImageFont.FreeTypeFont] = {}

    @staticmethod
    def _notdef_signature(font: ImageFont.FreeTypeFont) -> bytes:
        canvas = Image.new("L", (32, 32), 0)
        ImageDraw.Draw(canvas).text((2, 2), "\ue123", font=font, fill=255)
        return canvas.tobytes()

    def _signature(self, font: ImageFont.FreeTypeFont, ch: str) -> bytes:
        canvas = Image.new("L", (32, 32), 0)
        ImageDraw.Draw(canvas).text((2, 2), ch, font=font, fill=255)
        return canvas.tobytes()

    def font_for(self, ch: str, bold: bool) -> ImageFont.FreeTypeFont:
        key = (ch, bool(bold))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        base = self.bold if bold and self.bold is not None else self.regular
        chosen = base
        if self._signature(base, ch) == self._missing:
            for fallback in self.fallbacks:
                if self._signature(fallback, ch) != self._missing:
                    chosen = fallback
                    break
        self._cache[key] = chosen
        return chosen

    @property
    def cell_width(self) -> int:
        return round(self.regular.getlength("M"))

    @property
    def line_height(self) -> int:
        ascent, descent = self.regular.getmetrics()
        return round((ascent + descent) * LINE_SPACING)


def _blend(
    colour: tuple[int, int, int], background: tuple[int, int, int], alpha: float
) -> tuple[int, int, int]:
    return tuple(round(b + (c - b) * alpha) for c, b in zip(colour, background))


def render_window(rows: list[list[Cell]], title: str, fonts: FontBook) -> Image.Image:
    """Paint a terminal session into a window frame."""

    columns = max((len(row) for row in rows), default=1)
    cell_w = fonts.cell_width
    line_h = fonts.line_height

    width = columns * cell_w + WINDOW_PADDING * 2
    height = len(rows) * line_h + WINDOW_PADDING * 2 + TITLEBAR_HEIGHT

    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, 0, width, TITLEBAR_HEIGHT], fill=TITLEBAR)
    radius = 6
    for offset, colour in enumerate(((255, 95, 87), (254, 188, 46), (39, 200, 64))):
        cx = 18 + offset * 20
        draw.ellipse(
            [cx - radius, TITLEBAR_HEIGHT // 2 - radius, cx + radius, TITLEBAR_HEIGHT // 2 + radius],
            fill=colour,
        )
    title_font = fonts.title or fonts.regular
    try:
        title_width = draw.textlength(title, font=title_font)
    except AttributeError:  # pragma: no cover - very old Pillow
        title_width = fonts.regular.getlength(title)
    draw.text(
        ((width - title_width) / 2, TITLEBAR_HEIGHT / 2 - 7),
        title,
        font=title_font,
        fill=(139, 148, 158),
    )
    draw.line([(0, TITLEBAR_HEIGHT), (width, TITLEBAR_HEIGHT)], fill=BORDER)

    baseline_offset = TITLEBAR_HEIGHT + WINDOW_PADDING
    for row_index, row in enumerate(rows):
        y = baseline_offset + row_index * line_h
        for column_index, cell in enumerate(row):
            x = WINDOW_PADDING + column_index * cell_w
            if cell.bg is not None:
                draw.rectangle([x, y, x + cell_w, y + line_h], fill=cell.bg)
            colour = cell.fg or FOREGROUND
            if cell.dim:
                colour = _blend(colour, BACKGROUND, 0.55)
            draw.text(
                (x, y + 2), cell.ch, font=fonts.font_for(cell.ch, cell.bold), fill=colour
            )

    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, width - 1, height - 1], radius=10, fill=255)
    rounded = Image.new("RGB", (width, height), BACKGROUND)
    rounded.paste(image, (0, 0), mask)
    ImageDraw.Draw(rounded).rounded_rectangle(
        [0, 0, width - 1, height - 1], radius=10, outline=BORDER
    )
    return rounded


def prompt_line(command: str) -> list[Cell]:
    """`$ command`, styled the way a shell would style it."""

    return [Cell("$", PROMPT, None, True)] + [
        Cell(" " + chunk, FOREGROUND, None, False) for chunk in [command]
    ]


def session(title: str, blocks: list[tuple[str, str]], fonts: FontBook) -> Image.Image:
    """Build a window from a list of (command, output) pairs."""

    rows: list[list[Cell]] = []
    for index, (command, output) in enumerate(blocks):
        if index:
            rows.append([])
        rows.append(prompt_line(command))
        if output:
            rows.extend(parse_ansi(output.rstrip("\n")))
    return render_window(rows, title, fonts)


def run(cmd: list[str], cwd: Path, env_extra: dict[str, str] | None = None) -> str:
    """Run a real command and return exactly what it printed."""

    import os

    env = dict(os.environ)
    env["FORCE_COLOR"] = "1"
    env["COLUMNS"] = "200"
    env.pop("NO_COLOR", None)
    if env_extra:
        env.update(env_extra)
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    return completed.stdout + completed.stderr
