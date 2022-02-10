import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import strip


def to_xy(d):
    t, r = d[0, :], d[1, :]
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.stack([x, y])


def to_polar(d):
    x, y = d[0, :], d[1, :]
    theta = np.arctan(y / x)
    r = np.hypot(x, y)
    r = np.where(x > 0, r, -r)
    return np.stack([theta, r])


def alternating_mask(pos_distances, stripes):
    mask = np.zeros_like(pos_distances)
    for (b, e) in stripes:
        mask += np.logical_and(pos_distances >= b, pos_distances < e).astype(int)
    return mask.astype(bool)


if __name__ == "__main__":
    n = 1000
    np.random.seed(999)

    d = np.random.randn(2, n)

    # std = 1
    # s0 = np.random.standard_t(std, size=n)
    # s1 = np.random.standard_t(std, size=n)
    # d = np.stack([s0, s1])

    ############
    polar_d = to_polar(d)
    radius = np.abs(polar_d[1])

    step = 0.5
    stripes = [(i, i + step) for i in range(4)]
    mask = alternating_mask(radius, stripes)
    pos_d = d[:, mask]
    neg_d = d[:, ~mask]
    plt.figure(figsize=(5, 5.05))
    point_size = 10
    point_alpha = 0.7
    plt.scatter(*pos_d, s=point_size, alpha=point_alpha)
    plt.scatter(*neg_d, s=point_size, alpha=point_alpha)

    boundary = 3.5
    plt.xlim(-boundary, boundary)
    plt.ylim(-boundary, boundary)
    plt.axis("off")
    plt.show()

    ############
    radius = np.random.rand(n) * 3.5
    thetas = np.random.randn(n) * 0.95 + np.pi / 4.0
    polar_d = np.stack([thetas, radius])
    d = to_xy(polar_d)
    mask = alternating_mask(np.abs(radius), stripes)
    pos_d = d[:, mask]
    neg_d = d[:, ~mask]
    plt.figure(figsize=(5, 5.05))
    point_size = 10
    point_alpha = 0.7
    plt.scatter(*pos_d, s=point_size, alpha=point_alpha)
    plt.scatter(*neg_d, s=point_size, alpha=point_alpha)

    boundary = 3.5
    plt.xlim(-boundary, boundary)
    plt.ylim(-boundary, boundary)
    plt.autoscale(False)
    plt.axis("off")
    plt.show()

