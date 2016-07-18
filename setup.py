from setuptools import setup

setup(
    name='plenopy',
    version='0.0.0',
    description='View and work on plenoscope events',
    url='',
    author='Sebastian Mueller, Max L. Ahnen, Dominik Neise',
    author_email='sebmuell@phys.ethz.ch',
    license='MIT',
    packages=[
        'plenopy',
        ],
    install_requires=[
        'numpy',            # in anaconda
    ],
    entry_points={'console_scripts': [
        'mctPlenoscopePlotLixelStatistics = plenopy.main.plot_lixel_statistics:main',
    ]},
    zip_safe=False
)
