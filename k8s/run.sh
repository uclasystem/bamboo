#!/bin/bash

OPT=$1
if [[ ! -z $OPT ]]; then
	if [[ $OPT == clean ]]; then
		kubectl delete daemonset encoder -n elastic-job
		#kubectl delete --all pods -n elastic-job
		kubectl delete -f ~/projects/project-pactum/code/project-pactum/external/elastic/kubernetes/config/samples/etcd.yaml
		exit
	fi
fi

kubectl delete daemonset encoder -n elastic-job
kubectl delete -f ~/projects/project-pactum/code/end-to-end/project-pactum/external/elastic/kubernetes/config/samples/etcd.yaml
kubectl apply -f ~/projects/project-pactum/code/end-to-end/project-pactum/external/elastic/kubernetes/config/samples/etcd.yaml
sleep 10
kubectl apply -f ~/projects/project-pactum/code/end-to-end/project-pactum/k8s/encoder-daemonset.yaml
