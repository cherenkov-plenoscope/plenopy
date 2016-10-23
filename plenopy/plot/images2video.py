import os
import subprocess
import tempfile


def images2video(image_path, output_path ,frames_per_second=25):
    """
    Writes an h264 mp4 video to the output_path using the images found in the 
    image_path. The image sequence is given via a template path e.g. 
    'my/path/image_%06d.png'

    Parameters
    ----------
    image_path          Path to the input imaege sequence e.g. 'video_%06d.png'

    output_path         Path to write the final movie to

    frames_per_second   Number of frames per second in video
    """
    outpath = os.path.splitext(output_path)[0]
    stdout = open(outpath+'_stdout.txt', 'w')
    stderr = open(outpath+'_stderr.txt', 'w')
    rc = subprocess.call([
            'avconv',
            '-y',  # force overwriting of existing output file
            '-framerate', str(int(frames_per_second)),
            '-f', 'image2',
            '-i', image_path,
            '-c:v', 'h264',
            #'-s', '1260x1080', # sample images down to FullHD 1080p
            '-crf', '23',  # high quality 0 (best) to 53 (worst)
            '-crf_max', '25',  # worst quality allowed
            outpath+'.mp4'],
        stdout=stdout,
        stderr=stderr)
    stderr.close()
    stdout.close()
    return rc