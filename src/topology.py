from ripser import ripser

from .utils import obtain_points_for_each_label


def compute_homology(data, maxdim=2, subsample_size=1000, **kwargs):
    return ripser(data, maxdim=maxdim, n_perm=subsample_size, **kwargs)


def topological_complexity(
    X, labels=None, maxdim=1, subsample_size=1000, drop_zeroth=True, scale=None
):
    if labels is not None:
        data = obtain_points_for_each_label(X, labels)
    else:
        data = {-1: X}
    b_numbers = dict()
    for label, X in data.items():
        res = compute_homology(X, maxdim, subsample_size)["dgms"]
        if drop_zeroth:
            res = res[1:]
        if scale is not None:
            if isinstance(scale, int):
                res_at_scale = [
                    gen
                    for hom in res
                    for gen in hom
                    if gen[0] <= scale and scale < gen[1]
                ]
                b_numbers[label] = len(res_at_scale)
            if isinstance(scale, list):
                b_numbers[label] = dict()
                for s in scale:
                    res_at_scale = [
                        gen for hom in res for gen in hom if gen[0] <= s and s < gen[1]
                    ]
                    b_numbers[label][s] = len(res_at_scale)
        else:
            b_numbers[label] = sum([hom.shape[0] for hom in res])
    return b_numbers
