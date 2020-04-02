import tarfile
import tempfile
import os
from .Event import Event


def read_from_tar(path, light_field_geometry):
    with tempfile.TemporaryDirectory(suffix="plenopy_event") as tmp:
        tf = tarfile.open(path)
        event_basename = os.path.basename(path)
        if event_basename.endswith('.tar'):
            event_basename = event_basename.strip('.tar')
        tmp_event_path = os.path.join(tmp, event_basename)
        tf.extractall(tmp_event_path)
        event = Event(
            path=tmp_event_path,
            light_field_geometry=light_field_geometry)
    return event
