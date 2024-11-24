import numpy as np


def create_ring(center, inner_radius, outer_radius, num_points):
    """
    Create a ring with a given center, inner radius, and outer radius containing num_points.
    """
    angles = np.linspace(0, 2 * np.pi, num_points)
    radii = np.sqrt(np.random.rand(num_points) * (outer_radius**2 - inner_radius**2) + inner_radius**2)
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return np.vstack((x, y)).T


def create_torus(center, major_radius, minor_radius, num_points):
    """
    Create a torus with a given center, major radius, and minor radius containing num_points.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi = np.linspace(0, 2 * np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.flatten()
    phi = phi.flatten()

    x = center[0] + (major_radius + minor_radius * np.cos(theta)) * np.cos(phi)
    y = center[1] + (major_radius + minor_radius * np.cos(theta)) * np.sin(phi)
    z = center[2] + minor_radius * np.sin(theta)

    return np.vstack((x, y, z)).T


def create_sphere(center, radius, num_points):
    """
    Create a sphere with a given center and radius containing num_points.
    """
    theta = np.linspace(0, 2 * np.pi, num_points)
    phi = np.linspace(0, np.pi, num_points)
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.flatten()
    phi = phi.flatten()

    x = center[0] + radius * np.cos(theta) * np.sin(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(phi)

    return np.vstack((x, y, z)).T
