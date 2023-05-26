# -speed 3 -color_primaries 1 -color_trc 1 -colorspace 1 -movflags +faststart

ffmpeg -y -i "$1" -vf "scale='min(1600,iw)':-2" -vcodec libvpx -crf 48 -b:v 0 -threads 4 -cpu-used 4 -pix_fmt yuv420p -an "$2" < /dev/null
