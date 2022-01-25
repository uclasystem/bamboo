#!/bin/bash

version=$($(dirname "$0")/version.sh)
echo "${version}" > "$(dirname "$0")/.dockerversion"
docker build -t project-pactum:latest -t "project-pactum:${version%+*}" .
rm "$(dirname "$0")/.dockerversion"
