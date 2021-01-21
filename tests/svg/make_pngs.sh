#!/usr/bin/env bash

Inkscape='/Applications/Inkscape.app/Contents/MacOS/inkscape'
SRC=tests/svg
DST=tests/images

# shellcheck disable=SC2035
sources=$(cd $SRC && echo *.svg)

for file_name in $sources; do
    echo $Inkscape -o "$DST/${file_name%.svg}.png" "$SRC/$file_name"
    $Inkscape -o "$DST/${file_name%.svg}.png" "$SRC/$file_name"
done
