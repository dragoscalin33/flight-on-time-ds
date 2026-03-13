"""
Geodesic distance calculation using the Haversine formula.
"""
import numpy as np


def haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Vectorised Haversine distance (km) between two sets of coordinates.

    Parameters
    ----------
    lat1, lon1 : array-like  — Origin coordinates (degrees).
    lat2, lon2 : array-like  — Destination coordinates (degrees).

    Returns
    -------
    np.ndarray — Distance in kilometres, rounded to 2 decimals.
    """
    R = 6_371  # Earth radius in km

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    )
    distance = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return np.round(distance, 2)
