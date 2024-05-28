#! /bin/bash

for dir in $(ls ./)
do
	python /media/yons/U31/ycgong/retdec-v4.0-ubuntu-64b/retdec/bin/retdec-decompiler.py   $dir
done
