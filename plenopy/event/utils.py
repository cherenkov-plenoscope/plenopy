from ..tools import HeaderRepresentation as hr


def event_type_from_header(plenoscope_event_header):
    peh = plenoscope_event_header
    hr.assert_shape_is_valid(peh)
    hr.assert_marker_of_header_is(peh, 'PEVT')
    if peh[2-1] == 0.0:
        return 'OBSERVATION'
    elif peh[2-1] == 1.0:
        return 'SIMULATION'
    else:
        return 'unknown: '+str(peh[2-1])


def trigger_type_from_header(plenoscope_event_header):
    peh = plenoscope_event_header
    hr.assert_shape_is_valid(peh)
    hr.assert_marker_of_header_is(peh, 'PEVT')
    if peh[3-1] == 0.0:
        return 'SELF_TRIGGER'
    elif peh[3-1] == 1.0:
        return 'EXTERNAL_RANDOM_TRIGGER'
    elif peh[3-1] == 2.0:
        return 'EXTERNAL_TRIGGER_BASED_ON_AIR_SHOWER_SIMULATION_TRUTH'
    else:
        return 'unknown: '+str(peh[3-1])


def short_info(event):
    if event.type == "SIMULATION":
        if (event.trigger_type ==
            "EXTERNAL_TRIGGER_BASED_ON_AIR_SHOWER_SIMULATION_TRUTH"):
            return event.simulation_truth.event.short_event_info()
        elif event.trigger_type == "EXTERNAL_RANDOM_TRIGGER":
            return 'Extrenal random trigger, no air shower'
        else:
            return 'Simulation, but trigger type is unknown: '+str(
                event.trigger_type)
    elif event.type == "OBSERVATION":
        return 'Observation'
    else:
        return 'unknown event type: '+str(event.type)
