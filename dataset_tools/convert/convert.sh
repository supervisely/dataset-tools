#!/usr/bin/env bash

INPUT_FOLDER="${1:-..}"

find "$INPUT_FOLDER" -type d -name node_modules -prune -o -type f \( -iname *.mp4 -o -iname *.gif \) -print0 | while read -d $'\0' file
do
  DIR="$(dirname "$file")"
  PARENT_DIR="$(dirname "$DIR")"
  DST_DIRNAME="converted"

  NAME="$(basename "$file")"
  NEW_NAME="${NAME%.*}"

  NEW_DIR="${PARENT_DIR}/${DST_DIRNAME}"
  if [ ! -d "$NEW_DIR" ]; then
    mkdir "$NEW_DIR"
  fi

  if [ ! -f "${NEW_DIR}/${NEW_NAME}.mp4" ]; then
    echo "Converting: $file -> ${NEW_DIR}/${NEW_NAME}.mp4"
    ./mp4.sh "$file" "${NEW_DIR}/${NEW_NAME}.mp4"
  fi

  if [ ! -f "${NEW_DIR}/${NEW_NAME}.webm" ]; then
    echo "Converting: $file -> ${NEW_DIR}/${NEW_NAME}.webm"
    ./webm.sh "$file" "${NEW_DIR}/${NEW_NAME}.webm"
  fi
done