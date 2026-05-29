export HF_HOME="/local/scratch2/carlyn.1/hf_cache"

GUIDANCE_SCALE=1.5
START_STEP=600
CELL_SIZE=16
AREA=0.3

WANDB_MODE=online HYDRA_FULL_ERROR=1 python src/main.py \
    --config-name imagenet_cava_target_to_guide \
    exp.target_id=340 \
    exp.guide_id=339 \
    guidance.scale.value=$GUIDANCE_SCALE \
    inpainter.subconfig.start_step=$START_STEP \
    explainer.cell_size=$CELL_SIZE \
    explainer.area=$AREA