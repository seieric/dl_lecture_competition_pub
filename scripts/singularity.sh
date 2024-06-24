#!/bin/sh
singularity exec --nv --bind `pwd` singularity/container.sif python3 main.py