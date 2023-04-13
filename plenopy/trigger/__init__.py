"""
A simple sum-trigger
--------------------

Individual read-out-channels (light-field-cells) are added up and
compared against a threshold.

Here we sum the light-field-cells to create images refocused to certain
object-distances.

The total trigger-system can have multiple of these refocused images.

The pattern of light-field-cells to be added up is stored in the
trigger-geometry, which is estimated from the ligth-field-geometry.
"""
from . import io
from . import geometry
from . import estimate
from . import region_of_interest
from . import utils
