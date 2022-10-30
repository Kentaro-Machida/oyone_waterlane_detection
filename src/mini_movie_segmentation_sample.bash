MP4_PATH="../raw_data/9_6/2022_9_3_16_54/trimed.mp4"
MINI_STATE_DICT_PATH="../trained_models/mini_956_dataset/Unet-Mobilenet_v2_mIoU-0.886.pt"
OUT_PATH="../samples/output_movie/mini_segmented.mp4"
MODEL_SIZE="mini"
WIDTH=224
HEIGHT=224
python movie_segmentation.py --mp4_path=$MP4_PATH --state_dict=$MINI_STATE_DICT_PATH \
--model_size=$MODEL_SIZE --width=$WIDTH --height=$HEIGHT -save=$OUT_PATH