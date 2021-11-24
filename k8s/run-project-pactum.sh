#!/bin/bash

IP=${1}
ID=encoder${2}

if [[ -z $IP ]]; then
	echo "Must provide IP address of etcd"
	exit
fi

python -m project_pactum.run --rdzv_backend=etcd --rdzv_endpoint=$IP:2379 --rdzv_id=$ID --nnodes=1:64 --nproc_per_node=1 --project-pactum --max-pipe-parallel-size=24 /workspace/external/deepspeed/DeepSpeedExamples/pipeline_parallelism/transformer.py --backend=nccl --redundancy_level=1 --steps=10 --deepspeed --deepspeed_config=/workspace/external/deepspeed/DeepSpeedExamples/pipeline_parallelism/transformer.json
