MP4_PATH="../raw_data/9_6/2022_9_3_16_54/trimed.mp4"
STATE_DICT_PATH="../trained_models/400_dataset/Unet-Mobilenet_v2_mIoU-0.939.pt"
OUT_PATH="../samples/output_movie/segmented.mp4"
python movie_segmentation.py --mp4_path=$MP4_PATH --state_dict=$STATE_DICT_PATH -save=$OUT_PATH