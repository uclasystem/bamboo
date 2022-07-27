#!/bin/bash


export NCCL_BLOCKING_WAIT=1

if [[ ${1} == 0 ]]; then
    echo -1 > ~/workspace/fail-checker/fail-file && rm ~/workspace/rendezvous/*
fi

python -u transformer.py --deepspeed_config=transformer.json -a 172.31.14.92 -sa 172.31.4.3 -s 4 -p 8 -rl 1 -ws 8 --debug -r ${1} ${@:2}