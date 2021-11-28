#!/bin/bash
# helper script for making SSH ports to sherlock nodes
# - p flag is for port number
# - n flag is for sherlock compute node

port=''
node=''

while getopts 'p:n:' flag; do
    case "${flag}" in
        p) port=${OPTARG};;
        n) node=${OPTARG};;
    esac
done


output = $(ssh -L $port:localhost:$port sherlock ssh -L $port:localhost:$port -N $node &)

exit



