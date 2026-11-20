#!/bin/bash
# Fixed wrapper for Animal World Studio jobs on SLURM: "$1" = python, "$2" = job file (argument lists as JSON).
exec "$1" -m animalworld.studio.runjob "$2"
