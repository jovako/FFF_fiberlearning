from .utils import TrainValTest


def load_dataset(name: str, **kwargs) -> TrainValTest:
    if name in ["miniboone", "gas", "hepmass", "power"]:
        # note that the given train/val/test split is ignored and a fixed split is performed
        from .tabular import get_tabular_datasets
        return get_tabular_datasets(name=name, **kwargs)
    elif name in ["mnist_ds", "mnist", "mnist_split", "cifar10", "celeba", "saved_mnist"]:
        from .image import (get_celeba_datasets, get_cifar10_datasets,
                            get_mnist_datasets, get_mnist_downsampled, get_split_mnist, get_saved_mnist)
        if name == "mnist":
            return get_mnist_datasets(**kwargs)
        elif name == "mnist_ds":
            return get_mnist_downsampled(**kwargs)
        elif name == "saved_mnist":
            return get_saved_mnist(**kwargs)
        elif name == "mnist_ae":
            return get_ae_mnist(**kwargs)
        elif name == "mnist_split":
            return get_split_mnist(**kwargs)
        elif name == "cifar10":
            return get_cifar10_datasets(**kwargs)
        elif name == "celeba":
            return get_celeba_datasets(**kwargs)
    elif name in ["qm9", "dw4", "lj13", "lj55"]:
        from .molecular import (load_dw4_dataset, load_lj13_dataset,
                                load_lj55_dataset, load_qm9_dataset)
        if name == "qm9":
            return load_qm9_dataset(**kwargs)
        elif name == "dw4":
            return load_dw4_dataset(**kwargs)
        elif name == "lj13":
            return load_lj13_dataset(**kwargs)
        elif name == "lj55":
            return load_lj55_dataset(**kwargs)
    elif name.startswith("sbi_"):
        parts = name.split("_")
        taskname = "_".join(parts[1:])
        from .sbi import get_sbi_dataset
        return get_sbi_dataset(name=taskname, **kwargs)
    elif name == "special-orthogonal":
        from fff.data.special_orthogonal import make_so_data
        return make_so_data(**kwargs)
    elif name.startswith("torus_"):
        if name == "torus_protein":
            from .torus import get_torus_protein_dataset
            return get_torus_protein_dataset(**kwargs)
        elif name == "torus_rna":
            from .torus import get_torus_rna_dataset
            return get_torus_rna_dataset(**kwargs)
    elif name in ["fire", "flood", "quakes", "volcano"]:
        from .earth import get_earth_dataset
        return get_earth_dataset(name, **kwargs)
    elif name in ["moons_split"]:
        from .toy import get_split_moons
        return get_split_moons(**kwargs)
    else:
        from .toy import make_toy_data
        return make_toy_data(name, **kwargs)

    raise ValueError(f"Unknown dataset {name} (unreachable code!)")
