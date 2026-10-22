#!/bin/bash
set -e
set -x

PUSH_S3=false
PUSH_DH=false
PUSH_CR=false
HAS_OPTIONS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        s3)
            PUSH_S3=true
            HAS_OPTIONS=true
            shift
            ;;
        dh)
            PUSH_DH=true
            HAS_OPTIONS=true
            shift
            ;;
        cr)
            PUSH_CR=true
            HAS_OPTIONS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

docker build -f launch_runner.Dockerfile -t launch_runner:latest -t cr.selcloud.ru/neurolab/launch_runner:latest -t develorium/launch_runner:latest ..

if [ "$HAS_OPTIONS" = false ]; then
    [[ -f .docker_push ]] && docker push cr.selcloud.ru/neurolab/launch_runner:latest
fi

if [ "$PUSH_DH" = true ]; then
    docker push develorium/launch_runner:latest
fi

if [ "$PUSH_CR" = true ]; then
    docker push cr.selcloud.ru/neurolab/launch_runner:latest
fi

if [ "$PUSH_S3" = true ]; then
    echo "Creating launch_runner.tgz..."
    docker save launch_runner:latest | gzip > launch_runner.tgz
    
    echo "Uploading launch_runner.tgz to S3..."
    aws s3 cp launch_runner.tgz s3://neurolab/launch_runner.tgz
    
    rm launch_runner.tgz
fi

