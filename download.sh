#!/bin/bash

if [ "$1" = "" -o "$2" = "" ]; then
	echo "Usage: download <directory> <youtube channel name>"
	exit 1
fi

for vid in `./list-yt-channel.py "$2"`; do
	echo "https://youtu.be/$vid"
	yt-dlp --quiet --no-warnings -x https://youtu.be/$vid -o "$1/training/$vid"
	ffmpeg -v 1 -i "$1/training/$vid.m4a" "$1/training/$vid.wav"
	rm "$1/training/$vid.m4a"
done

