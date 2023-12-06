export MODEL_PATH="/home/ai04/data/models/diffusion/Prompt2MedImage"
export TRAIN_DATA_DIR="/home/ai04/data/datasets/malignant_crop_image"
export OUTPUT_DIR="/home/ai04/data/models/Textual_inversion/1128"
export NCCL_P2P_LEVEL=NVL
accelerate launch train_textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --train_data_dir=$TRAIN_DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<cancer>" --initializer_token="breast" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=10000 \
  --learning_rate=5.0e-04 --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=500 \
  --output_dir=$OUTPUT_DIR 