MP4_PATH="../raw_data/9_6/2022_9_3_16_54/trimed.mp4"
STATE_DICT_PATH="../trained_models/iriomote_train_224_5d/Unet-Mobilenet_v2_mIoU-0.941.pt"
OUT_PATH="../samples/output_movie/iriomote_train_224_5d.mp4"
MODEL_SIZE="normal"
WIDTH=224
HEIGHT=224
python movie_segmentation.py --mp4_path=$MP4_PATH --state_dict=$STATE_DICT_PATH \
--model_size=$MODEL_SIZE --width=$WIDTH --height=$HEIGHT -save=$OUT_PATH \
--quantization