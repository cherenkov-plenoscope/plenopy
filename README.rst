#######
plenopy
#######
|TestStatus| |PyPiStatus| |BlackStyle| |BlackPackStyle| |GplV3LicenseBadge|

Read, plot, and investigate the recorded (simulated) events of the atmospheric Cherenkov plenoscope. Plenopy reads the plenoscope's events written by the [merlict_development_kit](https://github.com/cherenkov-plenoscope/merlict_development_kit).

*******
install
*******

.. code-block::

    pip install git+https://github.com/cherenkov-plenoscope/plenopy/


***********
basic usage
***********

.. code-block:: python

    import plenopy as pl

    run = pl.Run('plenopy/plenopy/tests/resources/run.acp/')
    event = run[3]
    event.show()


|ImgExampleEvent|


.. |BlackStyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |TestStatus| image:: https://github.com/cherenkov-plenoscope/plenopy/actions/workflows/test.yml/badge.svg?branch=main
    :target: https://github.com/cherenkov-plenoscope/plenopy/actions/workflows/test.yml

.. |PyPiStatus| image:: https://img.shields.io/pypi/v/plenopy
    :target: https://pypi.org/project/plenopy

.. |BlackPackStyle| image:: https://img.shields.io/badge/pack%20style-black-000000.svg
    :target: https://github.com/cherenkov-plenoscope/black_pack

.. |GplV3LicenseBadge| image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
    :target: https://www.gnu.org/licenses/gpl-3.0

.. |ImgExampleEvent| image:: https://github.com/cherenkov-plenoscope/plenopy/blob/main/readme/example_event_show.png?raw=True
