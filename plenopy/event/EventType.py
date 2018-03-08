from ..tools import HeaderRepresentation

class EventType(object):
    """
    The Atmospheric Cherenkov Plenoscope event type.
    """
    def __init__(self, raw):
        """
        Parameters
        ----------
        raw         The raw 273 float32 array.
        """
        HeaderRepresentation.assert_shape_is_valid(raw)

        # Event Type
        if raw[  2-1] == 0.0:
            self.type = 'OBSERVATION'
        elif raw[  2-1] == 1.0:
            self.type = 'SIMULATION'
        else:
            self.type = 'unknown: '+str(raw[  2-1])

        # Trigger Type
        if raw[  3-1] == 0.0:
            self.trigger_type = 'SELF_TRIGGER'
        elif raw[  3-1] == 1.0:
            self.trigger_type = 'EXTERNAL_RANDOM_TRIGGER'
        elif raw[  3-1] == 2.0:
            self.trigger_type = 'EXTERNAL_TRIGGER_BASED_ON_AIR_SHOWER_SIMULATION_TRUTH'
        else:
            self.trigger_type = 'unknown: '+str(raw[  3-1])


    def __repr__(self):
        out = 'EventType('
        out+= str(self.type)+', Trigger: '+str(self.trigger_type)+')'
        return out
