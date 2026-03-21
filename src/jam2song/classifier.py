import numpy as np
from scipy.spatial.distance import cdist

from .models import Segment

SLOPE_THRESHOLD = 0.02  # normalized RMS per normalized-time unit


def classify(segments: list[Segment]) -> tuple[list[Segment], np.ndarray]:
    """
    Assign energy_tier and trend to each segment.
    Returns the modified segments and a pairwise spectral distance matrix.
    """
    if not segments:
        return segments, np.zeros((0, 0))

    energies = np.array([s.mean_energy for s in segments])
    brightnesses = np.array([s.mean_brightness for s in segments])
    densities = np.array([s.onset_density for s in segments])

    # Normalize onset_density to 0-1 across all segments
    d_lo, d_hi = densities.min(), densities.max()
    if d_hi - d_lo > 1e-10:
        densities_norm = (densities - d_lo) / (d_hi - d_lo)
    else:
        densities_norm = np.zeros_like(densities)

    # Energy tier via quartiles
    q25 = float(np.percentile(energies, 25))
    q75 = float(np.percentile(energies, 75))

    for i, seg in enumerate(segments):
        e = energies[i]
        if e <= q25:
            seg.energy_tier = "low"
        elif e >= q75:
            seg.energy_tier = "high"
        else:
            seg.energy_tier = "mid"

        slope = seg.energy_slope
        if slope > SLOPE_THRESHOLD:
            seg.trend = "rising"
        elif slope < -SLOPE_THRESHOLD:
            seg.trend = "falling"
        else:
            seg.trend = "steady"

    # Pairwise distance matrix on [energy, brightness, onset_density_norm]
    feature_matrix = np.column_stack([energies, brightnesses, densities_norm])
    dist_matrix = cdist(feature_matrix, feature_matrix, metric="euclidean")

    return segments, dist_matrix
