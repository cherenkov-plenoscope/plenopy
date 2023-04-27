import setuptools
import numpy
import os


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


with open("README.md", "r") as f:
    long_description = f.read()

setup_py_path = os.path.realpath(__file__)
setup_py_dir = os.path.dirname(setup_py_path)
extra_files = package_files(os.path.join(setup_py_dir, 'plenopy', 'tests'))

setuptools.setup(
    name='plenopy',
    version='0.2.0',
    description='View and work on plenoscope events',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/cherenkov-plenoscope/plenopy.git',
    author='Sebastian Achim Mueller, Max L. Ahnen, Dominik Neise',
    author_email='sebastian-achim.mueller@mpi-hd.mpg.de',
    license='GPL v3',
    packages=['plenopy'],
    package_data={'plenopy': extra_files + [os.path.join("trigger", "scripts", "*")]},
    install_requires=[
        'setuptools>=18.0',
        'cython',
        'joblib',
        'ray_voxel_overlap',
        'sebastians_matplotlib_addons',
    ],
    zip_safe=False,
    ext_modules=[
        setuptools.Extension(
            "plenopy.photon_stream.cython_reader",
            sources=[
                os.path.join("plenopy", "photon_stream", "cython_reader.pyx"),
                os.path.join("plenopy", "photon_stream", "reader.c")
            ],
            include_dirs=[numpy.get_include(), "plenopy"],
            language="c",
        ),
    ],
)
