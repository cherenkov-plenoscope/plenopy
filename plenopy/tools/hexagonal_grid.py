import numpy as np


def make_hexagonal_grid(outer_radius, spacing, inner_radius=0.0):
    hex_a = np.array([np.sqrt(3) / 2, 0.5]) * spacing
    hex_b = np.array([0, 1]) * spacing

    grid = []
    sample_radius = 2.0 * np.floor(outer_radius / spacing)
    for a in np.arange(-sample_radius, sample_radius + 1):
        for b in np.arange(-sample_radius, sample_radius + 1):
            cell_ab = hex_a * a + hex_b * b
            cell_norm = np.linalg.norm(cell_ab)
            if cell_norm <= outer_radius and cell_norm >= inner_radius:
                grid.append(cell_ab)
    return np.array(grid)
