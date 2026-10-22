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

docker build -f neurolab_source.Dockerfile \
  -t cr.selcloud.ru/neurolab/neurolab_source:latest \
  -t neurolab_source:latest \
  --build-arg CACHEBUST=$(date +%s) \
  --secret id=neurolab_deploy_key,src=/home/misha/.ssh/neurolab_deploy_key \
  --network=host .

if [ "$HAS_OPTIONS" = false ]; then
    [[ -f .docker_push ]] && docker push cr.selcloud.ru/neurolab/neurolab_source:latest
fi

if [ "$PUSH_DH" = true ]; then
    docker tag neurolab_source:latest develorium/neurolab_source:latest
    docker push develorium/neurolab_source:latest
fi

if [ "$PUSH_CR" = true ]; then
    docker push cr.selcloud.ru/neurolab/neurolab_source:latest
fi

if [ "$PUSH_S3" = true ]; then
    echo "Creating neurolab_source.tgz..."
    docker save neurolab_source:latest | gzip > neurolab_source.tgz
    
    echo "Uploading neurolab_source.tgz to S3..."
    aws s3 cp neurolab_source.tgz s3://neurolab/neurolab_source.tgz
    
    rm neurolab_source.tgz
fi

