from .data_utils import create_ring, create_torus
from .visualisations import scatterplot

from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn import datasets as dt
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader


from abc import ABC, abstractmethod


class Dataset:
    """
    A class to represent a dataset.

    ...

    Attributes:
    ----------
    X: np.ndarray
    y: np.ndarray
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, name="", homology=None):
        self.X = X
        self.y = y
        self.name = name
        self.n_points = self.X.shape[0]
        self.dim = self.X.shape[1]
        self.homology = homology
        self.tensor_dataset = self.create_tensor_dataset()

    def create_tensor_dataset(self):
        if isinstance(self.X, np.ndarray):
            return TensorDataset(
                torch.from_numpy(self.X).to(torch.float32),
                torch.LongTensor(self.y),
            )
        if isinstance(self.X, torch.Tensor):
            if not isinstance(self.y, torch.LongTensor):
                self.y = self.y.type(torch.LongTensor)
            return TensorDataset(self.X.to(torch.float32), self.y)

    def train_test_split(
        self,
        batch_size=-1,
        test_ratio=0.2,
        val=False,
        val_ratio=0.2,
        datatype="numpy",
        device=None,
        legacy=False,
    ):
        if legacy:
            data = np.hstack([self.X, self.y.reshape(-1, 1)])
            np.random.shuffle(data)
            if datatype == "torch":
                data = torch.Tensor(data)
                if device:
                    data = data.to(device)
            test_thres = int(self.n_points * test_ratio)
            test_x, test_y = data[test_thres:, :-1], data[test_thres:, -1]
            train_x, train_y = data[:-test_thres, :-1], data[:-test_thres, -1]

            if val:
                val_thres = int(self.n_points * val_ratio)
                val_x, val_y = train_x[val_thres:, :], train_y[val_thres:, :]
                train_x, train_y = train_x[:-val_thres, :], train_y[:-val_thres, :]
                return train_x, train_y, val_x, val_y, test_x, test_y

            return train_x, train_y, test_x, test_y
        if self.tensor_dataset:
            train_set, test_set = random_split(
                self.tensor_dataset, [1 - test_ratio, test_ratio]
            )
            train_loader = DataLoader(
                dataset=train_set,
                batch_size=len(train_set) if batch_size == -1 else batch_size,
                shuffle=True,
            )
            test_loader = DataLoader(
                dataset=test_set,
                batch_size=len(train_set) if batch_size == -1 else batch_size,
                shuffle=False,
            )
            return train_loader, test_loader

    def plot_data(self, save=False, color=None, **kwargs):
        if color is None:
            color = self.y
        if self.dim == 2:
            return scatterplot(
                x_coords=self.X[:, 0], y_coords=self.X[:, 1], color=color, save=save
            )
        if self.dim == 3:
            return scatterplot(
                x_coords=self.X[:, 0],
                y_coords=self.X[:, 1],
                z_coords=self.X[:, 2],
                color=color,
                save=save,
                dim=3,
                **kwargs,
            )


class Circles(Dataset):
    def __init__(self, n_samples=2000, n_circles_per_class=4, margin=3, noise=0.15):
        self.n_samples = n_samples
        self.n_circles_per_class = n_circles_per_class
        homology = {
            0: [self.n_circles_per_class, self.n_circles_per_class],
            1: [self.n_circles_per_class, self.n_circles_per_class],
        }
        super().__init__(
            *self.__generate_data(margin, noise), name="circles", homology=homology
        )

    def __generate_data(self, margin, noise):
        circles_in_a_row = int(np.sqrt(self.n_circles_per_class))
        data_x, data_y = dt.make_circles(n_samples=self.n_samples)
        for i in range(1, self.n_circles_per_class):
            new_data_x, new_data_y = dt.make_circles(n_samples=self.n_samples)
            margin_0, margin_1 = (
                i // circles_in_a_row + noise,
                i % circles_in_a_row + noise,
            )
            new_data_x[:, 0], new_data_x[:, 1] = (
                new_data_x[:, 0] + margin * margin_0,  # + np.random.normal(0, noise),
                new_data_x[:, 1] + margin * margin_1,  # + np.random.normal(0, noise),
            )
            data_x = np.vstack([data_x, new_data_x])
            data_y = np.hstack([data_y, new_data_y])
        return data_x, data_y


class Blobs(Dataset):
    def __init__(self, n_samples=25, n_features=2):
        self.n_samples = n_samples
        self.n_features = n_features

        super().__init__(
            *dt.make_blobs(
                n_samples=self.n_samples, n_features=self.n_features, centers=2
            ),
            name="blobs",
        )


class Classes(Dataset):
    def __init__(self, n_samples=25, n_features=2):
        self.n_samples = n_samples
        self.n_features = n_features

        super().__init__(
            *dt.make_classification(
                n_samples=self.n_samples,
                n_features=self.n_features,
                n_classes=2,
                n_redundant=0,
            ),
            name="blobs",
        )


class Sphere(Dataset):
    def __init__(self, n_samples=50):
        self.n_samples = n_samples
        sp = np.linspace(0, 2.0 * np.pi, num=self.n_samples)
        u = np.repeat(sp, self.n_samples)
        v = np.tile(sp, self.n_samples)
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        sphere = np.vstack([x, y, z]).T

        super().__init__(
            X=sphere,
            y=np.zeros(self.n_samples**2),
            name="sphere",
        )


class Tori(Dataset):
    def __init__(
        self,
        n_samples=150,
        shapeParam1=15,
        shapeParam2=2.5,
        shapeParam3=2.2,
        coordinates=[
            (3, 3, 3),
            (-3, -3, 3),
            (-3, 3, -3),
            (3, -3, -3),
            (-3, -3, -3),
            (3, 3, -3),
            (-3, 3, 3),
            (3, -3, 3),
            (0, 0, 0),
        ],
        radius=1,
        rng=0.5,
        visual=False,
    ):
        self.samples = n_samples
        self.shapeParam1 = shapeParam1
        self.shapeParam2 = shapeParam2
        self.shapeParam3 = shapeParam3
        self.coordinates = coordinates
        self.radius = radius
        self.visual = visual
        self.range = rng
        self.n_entanglements = len(coordinates)
        homology = {0: [9, 9], 1: [9, 9]}
        super().__init__(*self.__generate_data(), name="tori", homology=homology)

    def __draw_circle(self, r, center, n, rand=True):
        angles = np.linspace(start=0, stop=n, num=n) * (np.pi * 2) / n
        X = np.zeros(shape=(n, 2))
        X[:, 0] = np.sin(angles) * r
        X[:, 1] = np.cos(angles) * r

        if rand:
            return X + center + np.random.rand(n, 2) * r / self.shapeParam1
        else:
            return X + center

    def __gen_ring(self, center, flip, q=1.4, r=1):
        N_SAMPLES = self.samples
        X = np.zeros(shape=(2 * N_SAMPLES, 3))
        y = np.zeros(shape=(2 * N_SAMPLES,))

        X1 = self.__draw_circle(r=r, center=np.array((0, 0)), n=N_SAMPLES, rand=False)
        X2 = self.__draw_circle(r=r, center=np.array((0, 0)), n=N_SAMPLES, rand=False)

        X[0:N_SAMPLES, 0] = (
            (X1[:, 0]) * self.shapeParam2
            + np.random.uniform(low=-self.range, high=self.range, n_points=X1.shape[0])
            * q
        )
        X[0:N_SAMPLES, 1] = (
            (X1[:, 1]) * self.shapeParam2
            + np.random.uniform(low=-self.range, high=self.range, n_points=X1.shape[0])
            * q
        )
        X[0:N_SAMPLES, 2] = (
            np.random.uniform(low=-self.range, high=self.range, n_points=X1.shape[0])
            * q
        )

        X[N_SAMPLES : 2 * N_SAMPLES, 0] = (
            X2[:, 0] * self.shapeParam3
            + np.random.uniform(low=-self.range, high=self.range, n_points=X1.shape[0])
            * q
        )
        X[N_SAMPLES : 2 * N_SAMPLES, 1] = (
            X2[:, 1] * self.shapeParam3
            + np.random.uniform(low=-self.range, high=self.range, n_points=X1.shape[0])
            * q
        )
        X[N_SAMPLES : 2 * N_SAMPLES, 2] = (
            np.random.uniform(low=-self.range, high=self.range, n_points=X1.shape[0])
            * q
        )

        y[:] = flip
        y[0:N_SAMPLES] = flip

        X_total = X.copy() + np.array((self.shapeParam3, 0, 0))
        y_total = y.copy()

        X = np.zeros(shape=(2 * N_SAMPLES, 3))
        y = np.zeros(shape=(2 * N_SAMPLES,))

        X1 = self.__draw_circle(r=r, center=np.array((0, 0)), n=N_SAMPLES)
        X2 = self.__draw_circle(r=r, center=np.array((0, 0)), n=N_SAMPLES)

        X[0:N_SAMPLES, 0] = (
            (X1[:, 0]) * self.shapeParam2
            + np.random.uniform(low=-self.range, high=self.range, n_points=X1.shape[0])
            * q
        )
        X[0:N_SAMPLES, 2] = (
            (X1[:, 1]) * self.shapeParam2
            + np.random.uniform(low=-self.range, high=self.range, n_points=X1.shape[0])
            * q
        )
        X[0:N_SAMPLES, 1] = (
            np.random.uniform(low=-self.range, high=self.range, n_points=X1.shape[0])
            * q
        )

        X[N_SAMPLES : 2 * N_SAMPLES, 0] = (
            X2[:, 0] * self.shapeParam3
            + np.random.uniform(low=-self.range, high=self.range, n_points=X1.shape[0])
            * q
        )
        X[N_SAMPLES : 2 * N_SAMPLES, 2] = (
            X2[:, 1] * self.shapeParam3
            + np.random.uniform(low=-self.range, high=self.range, n_points=X1.shape[0])
            * q
        )
        X[N_SAMPLES : 2 * N_SAMPLES, 1] = (
            np.random.uniform(low=-self.range, high=self.range, n_points=X1.shape[0])
            * q
        )

        y[:] = 1 - flip
        y[0:N_SAMPLES] = 1 - flip

        X_total = np.concatenate((X_total, X), axis=0) + center
        y_total = np.concatenate((y_total, y), axis=0)

        return X_total, y_total

    def __generate_data(self, **kwargs):
        X_total_list = [None] * self.n_entanglements
        y_total_list = [None] * self.n_entanglements

        for i in range(self.n_entanglements):
            Xi, yi = self.__gen_ring(self.coordinates[i], i % 2, **kwargs)
            X_total_list[i] = Xi
            y_total_list[i] = yi

        X_total = np.concatenate(X_total_list, axis=0)
        y_total = np.concatenate(y_total_list, axis=0)

        max_abs_scaler = preprocessing.MaxAbsScaler()
        X = max_abs_scaler.fit_transform(X_total)

        return X, y_total


class Disks(Dataset):
    def __init__(
        self, random=False, grid_min=-9, grid_max=9, res=0.19, big_r=7, small_r=2.1, n=9
    ):
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.res = res
        self.random = random
        self.big_r = big_r
        self.small_r = small_r
        self.n = n
        super().__init__(*self.__generate_data(), name="disks")

    def __gen_grid(
        self,
        centers=[(0, 0), (8, 8)],
        radius=0.5,
        margin=0.5,
    ):
        x = np.arange(self.grid_min, self.grid_max, self.res)
        y = np.arange(self.grid_min, self.grid_max, self.res)
        xx, yy = np.meshgrid(x, y)
        if self.random:
            xx = xx + np.random.randn(xx.shape[0], xx.shape[1]) * 0.02
            yy = yy + np.random.randn(yy.shape[0], yy.shape[1]) * 0.02
        grid = np.dstack((xx, yy)).reshape(-1, 2)

        y = np.ones(shape=(len(grid), 2)) * 1

        for center in centers:
            for i, point in enumerate(grid):
                if np.linalg.norm(center - point) < radius + margin:
                    y[i, 0] = 3
                    y[i, 1] = 3

                if np.linalg.norm(center - point) < radius:
                    y[i, 0] = 0 * y[i, 0]
                    y[i, 1] = not y[i, 0]

            y = np.array(y)

            mask = y == 3
            label = y[:, 0]
            label = np.delete(label, np.where(mask[:, 0]))
            xx = list(grid[:, 0])
            yy = list(grid[:, 1])

            xx = np.delete(xx, np.where(mask[:, 0]))
            yy = np.delete(yy, np.where(mask[:, 0]))
            grid = np.ones(shape=(len(xx), 2))

            grid[:, 0] = xx
            grid[:, 1] = yy

            y = np.ones(shape=(len(xx), 2))
            y[:, 0] = label
            y[:, 1] = 1 - label

        return (grid, y[:, 0])

    def __generate_data(self):
        centers = [
            (
                np.cos(2 * np.pi / (self.n - 1) * x) * self.big_r,
                np.sin(2 * np.pi / (self.n - 1) * x) * self.big_r,
            )
            for x in range(0, self.n)
        ] + [0, 0]
        X, y = self.__gen_grid(
            centers=centers,
            radius=self.small_r,
        )

        return X, y


class NestedDataset(Dataset, ABC):
    def __init__(
        self,
        name: str,
        homology: Dict[int, List[int]] = None,
        base_n_points=2000,
        n_in_nest=3,
        n_in_row=2,
    ):
        self.base_n_points = base_n_points
        self.n_in_nest = n_in_nest
        self.n_in_row = n_in_row
        super().__init__(*self.__generate_data(), name=name, homology=homology)

    @abstractmethod
    def _generate_parameters_for_creating_data_object(
        self, i: int, j: int
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _create_data_object(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def __generate_data(self):
        data = []
        labels = []
        for i in range(self.n_in_row):
            for j in range(self.n_in_nest):
                parameters = self._generate_parameters_for_creating_data_object(i, j)
                data_object = self._create_data_object(**parameters)
                data.append(data_object)

                label = 0 if j % 2 == 0 else 1
                labels.append(np.full(data_object.shape[0], label, dtype=int))

        data = np.vstack(data)
        labels = np.concatenate(labels)

        return data, labels


class NestedRings(NestedDataset):
    def __init__(
        self,
        base_n_points=2000,
        n_in_row=2,
        n_in_nest=3,
        radius_factor=0.5,
        base_radius=10,
        noise=1.5,
    ):
        self.radius_factor = radius_factor
        self.base_radius = base_radius
        self.noise = noise
        homology = {
            0: [
                n_in_nest * n_in_row,
                n_in_nest * n_in_row,
            ],
            1: [
                n_in_nest * n_in_row,
                n_in_nest * n_in_row,
            ],
        }
        super().__init__(
            name="nested_rings",
            homology=homology,
            base_n_points=base_n_points,
            n_in_nest=n_in_nest,
            n_in_row=n_in_row,
        )

    def _generate_parameters_for_creating_data_object(
        self, i: int, j: int
    ) -> Dict[str, Any]:
        center = (
            np.array(
                [
                    (i * 2 - (self.n_in_row - 1)) * self.base_radius * self.noise,
                    self.noise * np.random.randn(),
                ]
            )
            + self.noise * np.random.rand()
        )
        outer_radius = self.base_radius * self.radius_factor**j
        inner_radius = (outer_radius + 3) * self.radius_factor
        num_points = self.base_n_points // (j + 1)
        return {
            "center": center,
            "outer_radius": outer_radius,
            "inner_radius": inner_radius,
            "num_points": num_points,
        }

    def _create_data_object(self, center, outer_radius, inner_radius, num_points):
        return create_ring(center, inner_radius, outer_radius, num_points)


class NestedTori(NestedDataset):
    def __init__(
        self,
        base_n_points=100,
        n_in_row=2,
        n_in_nest=3,
        radius_factor=0.5,
        base_radius=10,
        add_radius=2,
        noise=1.5,
    ):
        self.radius_factor = radius_factor
        self.base_radius = base_radius
        self.add_radius = add_radius
        self.noise = noise
        homology = {
            0: [
                n_in_nest * n_in_row,
                n_in_nest * n_in_row,
            ],
            1: [
                n_in_nest * n_in_row,
                n_in_nest * n_in_row,
            ],
        }
        super().__init__(
            name="nested_tori",
            homology=homology,
            base_n_points=base_n_points,
            n_in_nest=n_in_nest,
            n_in_row=n_in_row,
        )

    def _generate_parameters_for_creating_data_object(
        self, i: int, j: int
    ) -> Dict[str, Any]:
        center = np.array([(i * 2 - (self.n_in_row - 1)) * self.base_radius * 3, 0, 0])
        major_radius = self.base_radius * (self.radius_factor**j)
        minor_radius = self.add_radius * (self.radius_factor**j)
        num_points = self.base_n_points // (j + 1)
        return {
            "center": center,
            "major_radius": major_radius,
            "minor_radius": minor_radius,
            "num_points": num_points,
        }

    def _create_data_object(self, center, major_radius, minor_radius, num_points):
        return create_torus(center, major_radius, minor_radius, num_points)


class CurvesOnTorus(Dataset):
    def __init__(
        self,
        n_revolutions: int = 8,
        n_points: int = 1000,
        R: float = 5,
        r: float = 2,
        spiral_radius: float = 0.5,
        link_offset: float = np.pi,
        noise: float = 1.5,
    ):
        self.n_revolutions = n_revolutions
        self.R = (R,)
        self.r = r
        self.spiral_radius = spiral_radius
        self.noise = noise
        self.link_offset = link_offset
        self.n_points = n_points
        super().__init__(*self.__generate_data(), name="curves_on_torus")

    def __generate_data(self):
        t = np.linspace(0, 2 * np.pi * self.n_revolutions, self.n_points)
        labels = []

        data = np.vstack(
            [self.__generate_spiral(t, 0), self.__generate_spiral(t, self.link_offset)]
        )
        labels = np.concatenate([np.zeros(self.n_points), np.ones(self.n_points)])

        return data, labels

    def __generate_spiral(self, t, offset):
        phi = t + offset
        theta = t / self.n_revolutions

        u = np.random.uniform(0, 2 * np.pi, self.n_points)
        v = np.random.uniform(0, self.spiral_radius, self.n_points)

        x = (self.R + self.r * np.cos(phi) + v * np.cos(u) * np.cos(phi)) * np.cos(
            theta
        )
        y = (self.R + self.r * np.cos(phi) + v * np.cos(u) * np.cos(phi)) * np.sin(
            theta
        )
        z = self.r * np.sin(phi) + v * np.sin(u)

        return np.column_stack((x, y, z))


class BreastCancer(Dataset):
    def __init__(self):
        X, y = load_breast_cancer(return_X_y=True)
        name = "Breast cancer"
        super().__init__(X, y, name)
