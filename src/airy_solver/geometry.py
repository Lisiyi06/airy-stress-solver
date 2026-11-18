"""Geometric objects for rectangular Airy stress problems."""

from dataclasses import dataclass
import sympy as sp

L, h = sp.symbols("L h", positive=True)


LEFT = "left"
RIGHT = "right"
BOTTOM = "bottom"
TOP = "top"

EDGES = (LEFT, RIGHT, BOTTOM, TOP)


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned rectangular domain."""

    x_min: object
    x_max: object
    y_min: object
    y_max: object
    y_down: bool = False

    @property
    def top_y(self):
        return self.y_min if self.y_down else self.y_max

    @property
    def bottom_y(self):
        return self.y_max if self.y_down else self.y_min

    @property
    def length(self):
        return self.x_max - self.x_min

    @property
    def height(self):
        return self.y_max - self.y_min


def default_beam():
    """Symbolic beam, origin at left mid-height, y positive downward."""
    return Rectangle(0, L, -h/2, h/2, y_down=True)
