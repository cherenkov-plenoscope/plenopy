from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    name='plenopy',
    version='0.1.0',
    description='View and work on plenoscope events',
    url='',
    author='Sebastian Achim Mueller, Max L. Ahnen, Dominik Neise',
    author_email='sebmuell@phys.ethz.ch',
    license='MIT',
    packages=[
        'plenopy',
        'plenopy.corsika',
        'plenopy.event',
        'plenopy.idealized_plenoscope',
        'plenopy.image',
        'plenopy.light_field',
        'plenopy.light_field_geometry',
        'plenopy.main',
        'plenopy.photon_stream',
        'plenopy.plot',
        'plenopy.simulation_truth',
        'plenopy.tools',
    ],
    install_requires=[
        'numpy',            # in anaconda
    ],
    entry_points={'console_scripts': [
        'mctPlenoscopePlotLixelStatistics = plenopy.main.plot_lixel_statistics:main',
    ]},
    zip_safe=False,
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension(
            "plenopy.photon_stream.cython_reader",
            sources=[
                "plenopy/photon_stream/cython_reader.pyx", 
                "plenopy/photon_stream/reader.cpp"],
            include_dirs=[numpy.get_include(), "plenopy"],
            language="c++",
            extra_compile_args=['-std=c++0x']
        )
    ],
)
