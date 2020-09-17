"""
The photon-stream in the List-of-Photons (LOPF) representation

Format
======

    ASCII HEAD
    ----------
                        +-----+-----+-----+-----+
    Photon-stream line  |  P  |  H  |  S  |  |n |
                        +-----+-----+-----+-----+-----+
    Representation line |  L  |  O  |  P  |  H  |  |n |
                        +-----+-----+-----+-----+-----+
    Version line        |  V  |  1  |  |n |
                        +-----+-----+-----+

    Sensor HEAD
    -----------
                        +-----+-----+-----+-----+
    number_channels     |        uint32         |
                        +-----+-----+-----+-----+
    number_time_slices  |        uint32         |
                        +-----+-----+-----+-----+
    time_slice_duration |        float32        |
                        +-----+-----+-----+-----+

    Photons HEAD
    ------------
                        +-----+-----+-----+-----+-----+-----+-----+-----+
    number photons N    |                     uint64                    |
                        +-----+-----+-----+-----+-----+-----+-----+-----+
    Photons
    -------
    1st photon          +-----+-----+-----+-----+-----+
    channel, time_slice |         uint32        |uint8|
                        +-----+-----+-----+-----+-----+ .
                                                    .
                                                .
                                            .
                                        .
                                    .
                                .
                            .
                        .
    N-th photon         +-----+-----+-----+-----+-----+
    channel, time_slice |         uint32        |uint8|
                        +-----+-----+-----+-----+-----+

    END-OF-FILE

"""
import numpy as np


VERSION = 1

PHS_HEADER_LINE = "PHS\n".encode(encoding="ascii")
REPRESENTATION_LINE = "LOPH\n".encode(encoding="ascii")
VERSION_LINE = ("V{:d}\n".format(VERSION)).encode(encoding="ascii")


def _read(dtype, number, fileobj):
    size = number * np.dtype(dtype).itemsize
    return np.frombuffer(fileobj.read(size), dtype=dtype)


def assert_valid(phs):
    assert phs["photons"]["arrival_time_slices"].dtype == np.uint8
    assert phs["photons"]["channels"].dtype == np.uint32

    assert phs["sensor"]["number_channels"].dtype == np.uint32
    assert phs["sensor"]["number_time_slices"].dtype == np.uint32
    assert phs["sensor"]["time_slice_duration"].dtype == np.float32

    assert (
        phs["photons"]["arrival_time_slices"].shape
        == phs["photons"]["channels"].shape
    )


def write_photon_stream_to_file(phs, fileobj):
    """
    Write photon-stream (phs) in list-of-photons (loph) repr. to fileobj.
    """
    f = fileobj
    assert_valid(phs=phs)
    number_photons = np.uint64(phs["photons"]["channels"].shape[0])

    # HEADER
    # ======
    f.write(PHS_HEADER_LINE)
    f.write(REPRESENTATION_LINE)
    f.write(VERSION_LINE)

    # SENSOR
    # ======
    sen = phs["sensor"]
    f.write(sen["number_channels"].tobytes())
    f.write(sen["number_time_slices"].tobytes())
    f.write(sen["time_slice_duration"].tobytes())

    # PHOTONS
    # =======
    f.write(number_photons.tobytes())
    for ph in range(number_photons):
        f.write(phs["photons"]["channels"][ph].tobytes())
        f.write(phs["photons"]["arrival_time_slices"][ph].tobytes())


def read_photon_stream_from_file(fileobj):
    """
    Read photon-stream (phs) in list-of-photons (loph) repr. from fileobj.
    """
    f = fileobj
    phs = {"photons": {}, "sensor": {}}

    # HEADER
    # ======
    line = f.readline()
    assert line == PHS_HEADER_LINE
    line = f.readline()
    assert line == REPRESENTATION_LINE
    line = f.readline()
    assert line == VERSION_LINE

    # SENSOR
    # ======
    phs["sensor"]["number_channels"] = _read(np.uint32, 1, f)[0]
    phs["sensor"]["number_time_slices"] = _read(np.uint32, 1, f)[0]
    phs["sensor"]["time_slice_duration"] = _read(np.float32, 1, f)[0]

    # PHOTONS
    # =======
    num = _read(np.uint64, 1, f)[0]
    phs["photons"]["channels"] = np.zeros(num, dtype=np.uint32)
    phs["photons"]["arrival_time_slices"] = np.zeros(num, dtype=np.uint8)

    for ph in range(num):
        phs["photons"]["channels"][ph] = _read(np.uint32, 1, f)
        phs["photons"]["arrival_time_slices"][ph] = _read(np.uint8, 1, f)

    assert_valid(phs)
    return phs


def is_equal(phs_a, phs_b):
    for key in phs_a["sensor"]:
        if phs_a["sensor"][key] != phs_b["sensor"][key]:
            return False
    ap = phs_a["photons"]
    bp = phs_b["photons"]
    if not np.array_equal(ap["channels"], bp["channels"]):
        return False
    if not np.array_equal(
        ap["arrival_time_slices"], bp["arrival_time_slices"]
    ):
        return False
    return True
