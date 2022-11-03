LINESTRIP_DIR="../processed_data/labelme_annotations/956_datas"
POLYGON_DIR="../processed_data/polygons/956_datas_polygon2"
VOC_DIR="../processed_data/voc_datasets/956_dataset2_voc"
LABEL_TEXT="./labelme2voc/labels.txt"
python line2polygon.py --input_dir $LINESTRIP_DIR --output_dir $POLYGON_DIR
python ./labelme2voc/labelme2voc.py $POLYGON_DIR $VOC_DIR --labels $LABEL_TEXT