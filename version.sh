#!/bin/bash

version="$(cat $(dirname "$0")/VERSION)"
version_full="$(echo ${version} | cut -d '-' -f1 -s)"
if [ -z "${version_full}" ]; then
    version_full="${version}"
    version_pre=""
else
    version_pre="$(echo ${version} | cut -d '-' -f2 -s)"
fi

git_tag_exact_match=$(git describe --exact-match 2> /dev/null)
if [ $? -eq 0 ]; then
    if git diff --quiet 2> /dev/null; then
        if [ -n "${version_pre}" ]; then
            echo "Release version includes a pre-release" 1>&2
	    exit 1
        fi
        if [ "${version_full}" != "${git_tag_exact_match}" ]; then
	    echo "Release version doesn't match git tag: ${git_tag_exact_match}" 1>&2
	    exit 1
        fi
        echo "${version_full}"
        exit 0
    else
        if [ -z "${version_pre}" ]; then
            echo "Pre-release version needs a pre-release in VERSION file" 1>&2
	    exit 1
        fi
        if [ ${version_full} == ${git_tag_exact_match} ]; then
	    echo "Pre-release version matches a previous release" 1>&2
	    exit 1
        fi
        git_commit_id=$(git rev-parse --short=6 HEAD 2> /dev/null)
        if [ $? -ne 0 ]; then
	    exit 1
        fi
	echo "${version_full}-0.0+git.${git_commit_id}"
	exit 1
    fi
fi

git_tag_closest=$(git describe --abbrev=0 2> /dev/null)
if [ $? -eq 0 ]; then
    if [ ${version_full} == ${git_tag_closest} ]; then
	echo "Pre-release version matches a previous release" 1>&2
	exit 1
    fi
    git_commit_count=$(git rev-list --count "${git_tag_closest}..HEAD" 2> /dev/null)
    if [ $? -ne 0 ]; then
	exit 1
    fi
else
    git_commit_count=$(git rev-list --count HEAD 2> /dev/null)
fi

if [ -n "${git_commit_count}" ]; then
    if [ -z "${version_pre}" ]; then
	echo "Pre-release version needs a pre-release" 1>&2
	exit 1
    fi
    git_commit_id=$(git rev-parse --short=6 HEAD 2> /dev/null)
    if [ $? -ne 0 ]; then
	exit 1
    fi
    if git diff --quiet 2> /dev/null; then
        echo "${version_full}-${git_commit_count}+git.${git_commit_id}"
    else
	let "git_commit_count+=1"
        echo "${version_full}-${git_commit_count}+git.${git_commit_id}.dirty"
    fi
    exit 0
fi

if [ -z "${version_pre}" ]; then
    echo "${version_full}"
else
    echo "${version_full}-${version_pre}"
fi
