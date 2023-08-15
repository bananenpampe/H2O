#!/bin/bash

which python
rm /tmp/ipi_alchem_H2O

i-pi simulation.xml &

echo "Hello"
sleep 10s

echo "Current time: $(date +"%H:%M:%S")"

/Users/matthiaskellner/Documents/PhD/packages/develop/i-pi/bin/i-pi-py_driver -m lightning -a alchem_H2O -u -o example.ckpt,start.xyz 

