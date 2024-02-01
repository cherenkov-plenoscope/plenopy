import setuptools
import numpy
import os


with open("README.rst", "r", encoding="utf-8") as f:
    long_description = f.read()


with open(os.path.join("plenopy", "version.py")) as f:
    txt = f.read()
    last_line = txt.splitlines()[-1]
    version_string = last_line.split()[-1]
    version = version_string.strip("\"'")


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


setup_py_path = os.path.realpath(__file__)
setup_py_dir = os.path.dirname(setup_py_path)
extra_files = package_files(os.path.join(setup_py_dir, "plenopy", "tests"))


setuptools.setup(
    name="plenopy",
    version=version,
    description="View and work on plenoscope events",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/cherenkov-plenoscope/plenopy",
    author="Sebastian Achim Mueller, Max L. Ahnen, Dominik Neise",
    author_email="sebastian-achim.mueller@mpi-hd.mpg.de",
    packages=[
        "plenopy",
        "plenopy.trigger",
        "plenopy.photon_stream",
        "plenopy.corsika",
        "plenopy.plot",
        "plenopy.event",
        "plenopy.image",
        "plenopy.simulation_truth",
        "plenopy.light_field_geometry",
        "plenopy.tools",
        "plenopy.Tomography",
        "plenopy.Tomography.Image_Domain",
    ],
    package_data={
        "plenopy": extra_files + [os.path.join("trigger", "scripts", "*")]
    },
    install_requires=[
        "cython",
        "joblib",
        "thin_lens",
        "ray_voxel_overlap",
        "sebastians_matplotlib_addons",
    ],
    ext_modules=[
        setuptools.Extension(
            "plenopy.photon_stream.cython_reader",
            sources=[
                os.path.join("plenopy", "photon_stream", "cython_reader.pyx"),
                os.path.join("plenopy", "photon_stream", "reader.c"),
            ],
            include_dirs=[numpy.get_include(), "plenopy"],
            language="c",
        ),
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
