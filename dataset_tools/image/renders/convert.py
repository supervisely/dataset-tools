import subprocess
from subprocess import PIPE


def from_mp4_to_webm(src_path, dst_path, compression_level: int = 35):
    session = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-i",
            f"{src_path}",
            "-vf",
            "scale='min(1600,iw)':-2",
            "-vcodec",
            "libvpx-vp9",
            "-crf",
            f"{compression_level}",
            "-b:v",
            "0",
            "-threads",
            "4",
            "-cpu-used",
            "4",
            "-pix_fmt",
            "yuv420p",
            "-an",
            f"{dst_path}",
        ],
        stdout=PIPE,
        stderr=PIPE,
    )
    stdout, stderr = session.communicate()
    return dst_path


def compress_mp4(src_path, dst_path, compression_level: int = 35):
    session = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-i",
            f"{src_path}",
            "-vf",
            "scale='min(1600,iw)':-2",
            "-c:v",
            "libx264",
            "-crf",
            f"{compression_level}",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-an",
            f"{dst_path}",
        ],
        stdout=PIPE,
        stderr=PIPE,
    )
    stdout, stderr = session.communicate()
    return dst_path


def compress_png(src_path, dst_path, resolution: int = 720):
    session = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "0",
            "-i",
            f"{src_path}",
            "-filter:v",
            f"scale=-2:{resolution}",
            f"{dst_path}",
        ],
        stdout=PIPE,
        stderr=PIPE,
    )
    stdout, stderr = session.communicate()
    return dst_path
