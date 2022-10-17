# セマンティックセグメンテーションを用いた水田の水路検出
水上ドローンのカメラから得られた画像から水路を検出するプログラム。
## フォルダの説明
### low_camera_weed_detection
カメラを鉛直下向きにして、水中の雑草を検出するためのプログラムが置いてある。
* hsv_adjustment.py  
対象の画像を読み込み、HSV色空間の閾値をコマンドラインでリアルタイム操作し、うまく検出できるHSVの値を効率よく探索するためのツール。
* weed_in_water_detection  
動画を読み込み、ストックで使用したバウンディングボックス検出を利用して、水中の雑草を検出するためのプログラム。

### preprocess
アノテーションツール "labelme" を使用して、セマンティックセグメンテーション用のデータセットを作成するためのプログラムがあるディレクトリ。
* Line2Polygon2VOC.bash  
以下のコマンドを打てば、labelmeでlinestripとして、アノテーションされているデータを一度polygonのアノテーションに変更し、VOCデータセット形式に変化するという手順を1度に行うことができる。例は以下の通り。
~~~  
zsh Line2Polygon2VOC.bash  
~~~

* line2polygon.py  
コマンドで、指定ディレクトリを指定すると、labelmeでlinestripとして、アノテーションされている水路のデータをpolygonの水路のデータに変換できる。例は以下の通り。
~~~
python line2polygon.py --input_dir ../processed_data/labelme_annotations/113_datas_test --output_dir ../processed_data/polygons/113_datas_polygon
~~~

* labelme2voc/labelme2voc.py  
labelmeでのアノテーションをVOCデータセットに変換できる。デフォルトでは、labels.txtに書かれているクラスを参照する。例は以下の通り。
~~~
python ./labelme2voc/labelme2voc.py ../processed_data/polygons/113_datas_polygon ../processed_data/voc_datasets/113_dataset_voc --labels ./labelme2voc/labels.txt
~~~

* sapling.ipynb  
フレームレートが高すぎると、アノテーションの手間の割に似たデータばかりになってしまう。それを防ぐためにサンプリングを行うためのコード

### samples
* line_strip_annotation  
labelme を使用してline_stripとしてアノテーションし、出力されたjsonファイルがあるフォルダ。jsonファイル内には、アノテーションの情報と画像そのもののデータが含まれているため、ここから画像を復元可能。
* convert_to_polygon  
preprocess/line2polygon.py を使用してline_stripを水路のpolygonに変換させたjsonファイルがある。
* segmentation_voc  
preprocess/lebelme2voc/labelme2voc.py を使用して水路のpolygonからvocデータセットの形式に変換されたデータがあるディレクトリ。
### src  
実際のタスクを行うためのプログラムがあるディレクトリ。
* UNet.ipynb  
水路のセグメンテーションを行うUNetの学習、テストを行うためのプログラム。

* movie_segmentation.py  
学習済みモデルをロードして、動画を読み込み、セグメンテーションを行うためのプログラム。fpsも表示する。
