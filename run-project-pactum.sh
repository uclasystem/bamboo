#!/bin/bash

RDZV_IP=${1}
ID=encoder${2}
NUM_STAGES=${3}
NUM_STAGES=${NUM_STAGES:-4}

MODEL=${4}
MODEL=${MODEL:-'transformer'}

echo "ARGS $RDZV_ID $ID $NUM_STAGES $MODEL"

cmd="""export PROJECT_PACTUM_LOGGING_WARNING='etcd.client,etcd.lock,torch.distributed.distributed_c10d' \
	&& \
	export PYTHONPATH=/home/ubuntu/project-pactum:/home/ubuntu/workspace/bamboo/external/deepspeed:${PYTHONPATH} \
	&& \
	python -m project_pactum.run \
	--rdzv_backend=etcd-v2 \
	--rdzv_endpoint=$RDZV_IP:2379 \
	--rdzv_id=$ID \
	--nnodes=1:64 \
	--nproc_per_node=1 \
	--project-pactum \
	--max-pipe-parallel-size=24 \
	--default-num-stages=${NUM_STAGES} \
	/home/ubuntu/project-pactum/external/deepspeed/DeepSpeedExamples/pipeline_parallelism/${MODEL}.py \
	--backend=nccl \
	--redundancy_level=0 \
	${@:5} \
	--deepspeed \
	--deepspeed_config=/home/ubuntu/project-pactum/external/deepspeed/DeepSpeedExamples/pipeline_parallelism/${MODEL}.json"""

echo "RUNNING CMD $cmd"

eval $cmd