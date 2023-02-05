LINESTRIP_DIR="../processed_data/labelme_annotations/iriomote_test_labelme"
POLYGON_DIR="../processed_data/polygons/iriomote_test_polygon"
VOC_DIR="../processed_data/voc_datasets/iriomote_test_voc"
LABEL_TEXT="./labelme2voc/labels.txt"
python line2polygon.py --input_dir $LINESTRIP_DIR --output_dir $POLYGON_DIR
python ./labelme2voc/labelme2voc.py $POLYGON_DIR $VOC_DIR --labels $LABEL_TEXT