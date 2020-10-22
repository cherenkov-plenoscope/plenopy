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
import io
import tarfile
import os
from . import cython_reader

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
    n = 0
    f = fileobj
    assert_valid(phs=phs)
    number_photons = np.uint64(phs["photons"]["channels"].shape[0])

    # HEADER
    # ======
    n += f.write(PHS_HEADER_LINE)
    n += f.write(REPRESENTATION_LINE)
    n += f.write(VERSION_LINE)

    # SENSOR
    # ======
    sen = phs["sensor"]
    n += f.write(sen["number_channels"].tobytes())
    n += f.write(sen["number_time_slices"].tobytes())
    n += f.write(sen["time_slice_duration"].tobytes())

    # PHOTONS
    # =======
    n += f.write(number_photons.tobytes())
    for ph in range(number_photons):
        n += f.write(phs["photons"]["channels"][ph].tobytes())
        n += f.write(phs["photons"]["arrival_time_slices"][ph].tobytes())
    return n


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


def raw_sensor_response_to_photon_stream_in_loph_repr(
    raw_sensor_response, cherenkov_photon_ids
):
    raw = raw_sensor_response
    cer_ids = cherenkov_photon_ids
    (arrival_slices, lixel_ids,) = cython_reader.arrival_slices_and_lixel_ids(
        raw
    )
    phs = {}
    phs["sensor"] = {}
    phs["sensor"]["number_channels"] = raw.number_lixel
    phs["sensor"]["number_time_slices"] = raw.number_time_slices
    phs["sensor"]["time_slice_duration"] = raw.time_slice_duration

    phs["photons"] = {}
    phs["photons"]["channels"] = lixel_ids[cer_ids]
    phs["photons"]["arrival_time_slices"] = arrival_slices[cer_ids]
    assert_valid(phs=phs)
    return phs


class LopfTarReader:
    def __init__(self, path, id_num_digits=9):
        self.path = path
        self.tar = tarfile.open(path, "r|*")
        self._id_num_digits = id_num_digits
        assert self._id_num_digits > 0

    def __next__(self):
        info_tar = self.tar.next()
        if info_tar is None:
            raise StopIteration
        identity = int(info_tar.name[0 : self._id_num_digits])
        phs = read_photon_stream_from_file(
            fileobj=self.tar.extractfile(info_tar)
        )
        return identity, phs

    def __enter__(self):
        return self

    def __iter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return self.tar.close()

    def __repr__(self):
        out = "{:s}(path='{:s}')".format(self.__class__.__name__, self.path)
        return out


class LopfTarWriter:
    def __init__(self, path, id_num_digits=9):
        self.path = path
        self.tar = tarfile.open(path, "w")
        self._id_num_digits = id_num_digits
        assert self._id_num_digits > 0
        self._id_template_str = (
            "{:0" + str(self._id_num_digits) + "d}.phs.loph"
        )

    def add(self, identity, phs):
        with io.BytesIO() as buff:
            info = tarfile.TarInfo(self._id_template_str.format(identity))
            info.size = write_photon_stream_to_file(phs=phs, fileobj=buff)
            buff.seek(0)
            self.tar.addfile(info, buff)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.tar.close()
        return True

    def __repr__(self):
        out = "{:s}(path='{:s}')".format(self.__class__.__name__, self.path)
        return out


def concatenate_tars(in_paths, out_path):
    with tarfile.open(out_path, "w") as tarout:
        for part_path in in_paths:
            with tarfile.open(part_path, "r") as tarin:
                # woraround for https://bugs.python.org/issue29760
                try:
                    tarinfo = tarin.next()
                except OSError as e:
                    if e.errno == 22:
                        tarinfo = None
                    else:
                        raise e
                while tarinfo is not None:
                    tarout.addfile(tarinfo, tarin.extractfile(tarinfo))
                    tarinfo = tarin.next()


def split_into_chunks(loph_path, out_dir, chunk_prefix, num_events_in_chunk):
    """
    Helps with parallel computing
    """
    os.makedirs(out_dir, exist_ok=True)
    with LopfTarReader(path=loph_path) as run:
        chunk_count = 0
        has_events_left = True

        while has_events_left:

            chunk_path = os.path.join(
                out_dir,
                "{:s}{:09d}.tar".format(chunk_prefix, chunk_count)
            )

            with LopfTarWriter(path=chunk_path) as chunk:
                for ii in range(num_events_in_chunk):
                    try:
                        event = run.__next__()
                        identity, phs = event
                        chunk.add(identity=identity, phs=phs)
                    except StopIteration as stop:
                        has_events_left = False
            chunk_count += 1

def read_filter_write(in_path, out_path, identity_set):
    """
    Read in_path, and write event to out_path when idx is in identity_set
    """
    with LopfTarReader(in_path) as irun, LopfTarWriter(out_path) as orun:
        for event in irun:
            identity, phs = event
            if identity in identity_set:
                orun.add(identity=identity, phs=phs)
