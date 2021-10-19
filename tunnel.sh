#!/bin/bash

port=''
node=''

while getopts 'p:n:' flag; do
    case "${flag}" in
        p) port=${OPTARG};;
        n) node=${OPTARG};;
    esac
done


ssh -L $port:localhost:$port sherlock ssh -L $port:localhost:$port -N $node &

