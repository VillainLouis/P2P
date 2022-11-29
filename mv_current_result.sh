#!/bin/bash

# Move current result in ./clients and ./server to Results.
# Make a new filefold for current result according to the date inforation
cd ./Results
filefold_name='Record'--$(date +%Y_%m_%d--%H:%M:%S)
mkdir $filefold_name
cd $filefold_name
mv ../../server .
mv ../../clients . 
echo "Training Compele and Current Record Folder of server and clients has been moved to $filefold_name"