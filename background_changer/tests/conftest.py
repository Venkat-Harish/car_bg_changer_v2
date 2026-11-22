"""Shared helpers for the model-free tests."""

import cv2
import numpy as np
import pytest

from bg_changer import CarTable, clean_alpha

SRC_H, SRC_W = 2160, 3840


def side_car_alpha(h=SRC_H, w=SRC_W, body=(400, 700, 3400, 1700), tyres=((900, 1700), (2900, 1700)), r=230):
    """A side-view-like silhouette: a body box with two round tyres sticking out below it."""
    raw = np.zeros((h, w), np.float32)
    x1, y1, x2, y2 = body
    raw[y1:y2 - 150, x1:x2] = 1.0  # body; its bottom (sills) sits above the tyre bottoms
    for cx, cy in tyres:
        cv2.circle(raw, (cx, cy - r), r, 1.0, -1)
    return cv2.GaussianBlur(raw, (0, 0), 2)


@pytest.fixture(scope="session")
def table():
    return CarTable()


@pytest.fixture()
def side_car():
    """(image, CarMask) of a synthetic side view; every pixel distinct (noise)."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (SRC_H, SRC_W, 3), dtype=np.uint8)
    return image, clean_alpha(side_car_alpha(), edge_shrink=1)


@pytest.fixture()
def grey_plate():
    return np.full((1080, 1920, 3), 180, np.uint8)
