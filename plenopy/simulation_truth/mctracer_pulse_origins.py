"""
The magic-numbers of the mctracer plenoscope-simulation to describe the
origin of pulses in the raw-light-field-sensor-response.

- Positive integers represent the index of the photon from the input file,
  this is usually the index of the air-shower-photon in the KIT-CORSIKA
  input-file.

- Negative integers have special meanings for e.g. pulse-artifact or
  night-sky-background.
"""

MCTRACER_DEFAULT = -1
NIGHT_SKY_BACKGROUND = -100
PHOTO_ELECTRIC_CONVERTER_ACCIDENTAL = - 201
PHOTO_ELECTRIC_CONVERTER_CROSSTALK = - 202
