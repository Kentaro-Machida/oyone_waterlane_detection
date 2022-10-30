MP4_PATH="../raw_data/9_6/2022_9_3_16_54/trimed.mp4"
STATE_DICT_PATH="../trained_models/956_dataset/Unet-Mobilenet_v2_mIoU-0.939.pt"
OUT_PATH="../samples/output_movie/segmented.mp4"
MODEL_SIZE="normal"
WIDTH=480
HEIGHT=320
python movie_segmentation.py --mp4_path=$MP4_PATH --state_dict=$STATE_DICT_PATH \
--model_size=$MODEL_SIZE --width=$WIDTH --height=$HEIGHT -save=$OUT_PATH