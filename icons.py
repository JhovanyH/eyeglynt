"""
EyeGlynt - Icons
================
Draws the small icons used in the Figure 29 design (keyboard, flame,
speaker, erase, arrows, house) with Kivy's own drawing instructions.

Why draw them instead of using image files:
  - no image assets to download, license, or keep in sync between the
    laptop and the Pi;
  - they scale cleanly to any button size, since they are shapes,
    not pixels;
  - their colour can follow the button they sit on.

How it works: every icon is described inside a unit square, where
(0, 0) is the bottom-left corner and (1, 1) the top-right (Kivy's y
axis points UP). draw_icon() fits that square into the space given and
scales every coordinate to pixels.
"""

from math import cos, pi, sin

from kivy.graphics import Color, Ellipse, Line, Mesh, Rectangle, Triangle


def draw_icon(group, kind, x, y, w, h, color, bg_color):
    """Adds the instructions for icon `kind` to the InstructionGroup `group`.

    The icon is drawn as large as fits, centred, inside the box (x, y, w, h).
    `color` is the icon colour; `bg_color` is the colour of the button
    underneath, used for "cut-out" details such as the flame's inner
    tongue or the house's door.
    """
    side = min(w, h)
    ox = x + (w - side) / 2  # bottom-left corner of the square icon area
    oy = y + (h - side) / 2

    def p(u, v):
        """Converts a unit-square point (u, v) into pixel coordinates."""
        return ox + u * side, oy + v * side

    drawer = _ICONS.get(kind)
    if drawer is None:
        return
    group.add(Color(*color))
    drawer(group, p, side, color, bg_color)


# ---------------------------------------------------------------------
# Individual icons. Each receives the group to draw into, the p() point
# converter, the icon's side length in pixels, and the two colours.
# ---------------------------------------------------------------------

def _keyboard(group, p, side, color, bg):
    # Outline of the keyboard body.
    x0, y0 = p(0.05, 0.22)
    x1, y1 = p(0.95, 0.78)
    group.add(Line(rectangle=(x0, y0, x1 - x0, y1 - y0), width=max(1.0, side * 0.035)))
    # Three rows of keys.
    key_w, key_h = 0.075, 0.09
    for row_v in (0.62, 0.49, 0.36):
        for i in range(8):
            u = 0.12 + i * 0.0975
            kx, ky = p(u, row_v)
            group.add(Rectangle(pos=(kx, ky), size=(key_w * side, key_h * side)))
    # Space bar.
    sx, sy = p(0.28, 0.26)
    group.add(Rectangle(pos=(sx, sy), size=(0.44 * side, 0.06 * side)))


def _teardrop(p, cx, cy, half_w, half_h, n=48):
    """Vertices for a filled teardrop (pointed top, round bottom).

    Uses the curve x = sin(t) * sin(t/2), y = cos(t): at t = 0 it comes
    to a point (the flame's tip), and near t = pi it is round (the base).
    Returned in Mesh format: x, y, u, v for each vertex, with the centre
    first so the shape can be drawn as a triangle fan.
    """
    centre = p(cx, cy - half_h * 0.3)
    verts = [centre[0], centre[1], 0, 0]
    for i in range(n + 1):
        t = 2 * pi * i / n
        u = cx + half_w * sin(t) * sin(t / 2)
        v = cy + half_h * cos(t)
        px, py = p(u, v)
        verts += [px, py, 0, 0]
    return verts


# Outline of a flame with a tall centre tongue and a small tongue on
# each side, listed anticlockwise from the bottom. Every point can be
# "seen" from FLAME_CENTRE, which lets the shape be filled as a fan of
# triangles around that centre.
FLAME_OUTLINE = [
    (0.50, 0.05), (0.66, 0.08), (0.77, 0.17), (0.83, 0.29), (0.84, 0.41), (0.82, 0.52),
    (0.88, 0.74),                                   # right tongue tip
    (0.73, 0.61), (0.70, 0.71), (0.64, 0.81), (0.57, 0.89),
    (0.47, 0.98),                                   # main tip, leaning slightly left
    (0.42, 0.86), (0.35, 0.75), (0.30, 0.56),
    (0.15, 0.63),                                   # left tongue tip
    (0.15, 0.47), (0.17, 0.33), (0.23, 0.20), (0.33, 0.10),
]
FLAME_CENTRE = (0.50, 0.35)


def _fan(p, centre, outline):
    """Mesh vertices (x, y, u, v) for a shape filled as a triangle fan."""
    cx, cy = p(*centre)
    verts = [cx, cy, 0, 0]
    for u, v in outline + outline[:1]:  # repeat the first point to close the shape
        px, py = p(u, v)
        verts += [px, py, 0, 0]
    return verts


def _flame(group, p, side, color, bg):
    outer = _fan(p, FLAME_CENTRE, FLAME_OUTLINE)
    group.add(Mesh(vertices=outer, indices=list(range(len(outer) // 4)), mode="triangle_fan"))
    # Inner tongue in the button's colour gives the classic flame look.
    group.add(Color(*bg))
    inner = _teardrop(p, 0.5, 0.31, 0.17, 0.21)
    group.add(Mesh(vertices=inner, indices=list(range(len(inner) // 4)), mode="triangle_fan"))


def _speak(group, p, side, color, bg):
    # Head.
    hx, hy = p(0.12, 0.5)
    group.add(Ellipse(pos=(hx, hy), size=(0.3 * side, 0.3 * side)))
    # Shoulders: the top half of an ellipse (Kivy measures angles
    # clockwise from 12 o'clock, so -90..90 is the upper half).
    sx, sy = p(0.0, 0.06)
    group.add(Ellipse(pos=(sx, sy), size=(0.54 * side, 0.4 * side), angle_start=-90, angle_end=90))
    # Two sound waves, clear of the head.
    cx, cy = p(0.52, 0.65)
    lw = max(1.2, side * 0.05)
    for r in (0.16, 0.3):
        group.add(Line(circle=(cx, cy, r * side, 45, 135), width=lw, cap="round"))


def _erase(group, p, side, color, bg):
    # The backspace key shape: a box with a pointed left end, and an X.
    pts = []
    for u, v in ((0.06, 0.5), (0.3, 0.76), (0.92, 0.76), (0.92, 0.24), (0.3, 0.24)):
        pts += p(u, v)
    lw = max(1.2, side * 0.05)
    group.add(Line(points=pts, close=True, width=lw, joint="round"))
    group.add(Line(points=[*p(0.47, 0.37), *p(0.75, 0.63)], width=lw, cap="round"))
    group.add(Line(points=[*p(0.47, 0.63), *p(0.75, 0.37)], width=lw, cap="round"))


def _arrow_left(group, p, side, color, bg):
    group.add(Triangle(points=[*p(0.06, 0.5), *p(0.44, 0.84), *p(0.44, 0.16)]))
    x0, y0 = p(0.42, 0.36)
    group.add(Rectangle(pos=(x0, y0), size=(0.52 * side, 0.28 * side)))


def _shift(group, p, side, color, bg):
    group.add(Triangle(points=[*p(0.5, 0.92), *p(0.1, 0.5), *p(0.9, 0.5)]))
    x0, y0 = p(0.31, 0.1)
    group.add(Rectangle(pos=(x0, y0), size=(0.38 * side, 0.42 * side)))


def _home(group, p, side, color, bg):
    group.add(Triangle(points=[*p(0.06, 0.5), *p(0.5, 0.9), *p(0.94, 0.5)]))
    x0, y0 = p(0.18, 0.1)
    group.add(Rectangle(pos=(x0, y0), size=(0.64 * side, 0.42 * side)))
    # Door, in the button's colour.
    group.add(Color(*bg))
    dx, dy = p(0.41, 0.1)
    group.add(Rectangle(pos=(dx, dy), size=(0.18 * side, 0.24 * side)))


_ICONS = {
    "keyboard": _keyboard,
    "flame": _flame,
    "speak": _speak,
    "erase": _erase,
    "arrow_left": _arrow_left,
    "shift": _shift,
    "home": _home,
}