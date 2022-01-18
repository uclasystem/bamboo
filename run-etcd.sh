#!/bin/bash

etcd --data-dir ~/etcd \
    --enable-v2 \
    --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls http://0.0.0.0:2379 \
    --initial-cluster-state new
