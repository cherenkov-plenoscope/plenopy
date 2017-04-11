'''
Save plots of a plenoscope light field calibration. 
When the output_dir is not set, a plot folder is created in the input
calibration folder.

Usage:
    plenopyPlotLightFieldGeometry -i=INPUT_PATH [-o=OUTPUT_DIR]

Options:
    -o --output=OUTPUT_DIR     path to save the plots
    -i --input=INPUT_PATH      path to plenoscope calibration
'''
from __future__ import absolute_import, print_function, division
import docopt as do
import numpy as np
import os
from . import LightFieldGeometry
from . import PlotLightFieldGeometry


def main():
    try:
        arguments = do.docopt(__doc__)
        output_dir = arguments['--output']
        if output_dir is None:
            output_dir = os.path.join(arguments['--input'], 'plots')
            os.mkdir(output_dir)

        lfg = LightFieldGeometry(path=arguments['--input'])
        lfg_plotter = PlotLightFieldGeometry(
            light_field_geometry=lfg, 
            out_dir=output_dir)
        lfg_plotter.save()

    except do.DocoptExit as e:
        print(e)

if __name__ == '__main__':
    main()
