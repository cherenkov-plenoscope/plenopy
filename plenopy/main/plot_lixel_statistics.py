'''
Save plots of a plenoscope light field calibration. 
When the OUTPUT_PATH is not set, a plot folder is created in the input
calibration folder.

Usage:
    LixelStatisticsPlot -i=INPUT_PATH [-o=OUTPUT_PATH]

Options:
    -o --output=OUTPUT_PATH     path to save the plots
    -i --input=INPUT_PATH       path to plenoscope calibration
'''
from __future__ import absolute_import, print_function, division
import docopt as do
import numpy as np
import os
from ..light_field_geometry import LightFieldGeometry
from ..light_field_geometry import PlotLightFieldGeometry


def main():
    try:
        arguments = do.docopt(__doc__)
        output_path = arguments['--output']
        if output_path is None:
            output_path = os.path.join(arguments['--input'], 'plots')
            os.mkdir(output_path)

        lfg = LightFieldGeometry(path=arguments['--input'])
        lfg_plotter = PlotLightFieldGeometry(lfg, output_path)
        lfg_plotter.save()

    except do.DocoptExit as e:
        print(e)

if __name__ == '__main__':
    main()
