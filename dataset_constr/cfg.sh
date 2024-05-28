#! /bin/bash

for dir in $(ls ./)
do
   opt-10 -dot-cfg   $dir
done
