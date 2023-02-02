#!/bin/bash
export GPUS=1
export CPUS=8
export MEMORY=64000
export CLUSTER_ID=4
export MOUNT_POINT="/project_ghent"
export DOCKER_IMAGE="gitlab+deploy-token-8:iLvHxdZa4tyG4U5JXaCN@gitlab.ilabt.imec.be:4567/dml/gpulab/base:robosuite"

export DML_DEPLOY_TOKEN="gitlab+deploy-token-461"
export DML_DEPLOY_KEY="ignyXEzJ9WjYHQ5FDJEZ"

export CONFIG="dmc_pixels"
export AGENT="icm_dreamer"
export DOMAIN="panda"

#       git clone -b robosuite https://docker:MbKpnQbWwYyd9pywUEsq@gitlab.ilabt.imec.be/sferraro/choreographer.git && \

export COMMAND="bash -c \\\" \
        wandb login $WANDB_KEY && \
        git config --global --add safe.directory /project_ghent/sferraro/choreographer && \
        cd  /project_ghent/sferraro/choreographer && \
        git pull origin robosuite && \
        python3 dreamer_pretrain.py configs=$CONFIG agent=$AGENT domain=$DOMAIN \
        \\\""

echo "Submitting job"
cat gpulab-config.json | envsubst | gpulab-cli --password=$GPULAB_PASS submit --project=sferraro
