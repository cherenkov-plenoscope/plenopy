from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy
import os


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

setup_py_path = os.path.realpath(__file__)
setup_py_dir = os.path.dirname(setup_py_path)
extra_files = package_files(os.path.join(setup_py_dir, 'plenopy', 'tests'))

setup(
    name='plenopy',
    version='0.1.2',
    description='View and work on plenoscope events',
    url='https://github.com/cherenkov-plenoscope/plenopy.git',
    author='Sebastian Achim Mueller, Max L. Ahnen, Dominik Neise',
    author_email='sebmuell@phys.ethz.ch',
    license='GPL v3',
    packages=[
        'plenopy',
        'plenopy.corsika',
        'plenopy.event',
        'plenopy.image',
        'plenopy.light_field_geometry',
        'plenopy.tomography',
        'plenopy.tomography.image_domain',
        'plenopy.photon_stream',
        'plenopy.plot',
        'plenopy.simulation_truth',
        'plenopy.tools',
    ],
    package_data={'plenopy': extra_files},
    install_requires=[
        'numpy',            # in anaconda
        'joblib',
        'ray_voxel_overlap',
    ],
    entry_points={'console_scripts': []},
    zip_safe=False,
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            "plenopy.photon_stream.cython_reader",
            sources=[
                "plenopy/photon_stream/cython_reader.pyx",
                "plenopy/photon_stream/reader.cpp"
            ],
            include_dirs=[numpy.get_include(), "plenopy"],
            language="c++",
            extra_compile_args=['-std=c++0x']
        ),
    ],
)
