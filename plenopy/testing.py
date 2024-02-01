from importlib import resources as importlib_resources


def pkg_dir():
    return importlib_resources.files("plenopy")
