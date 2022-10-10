import json
import copy
import os
import glob
import argparse

def line2polygon(line_json_path, include_front:bool)->dict:
    """
    convert linestrip data to polygon data annoteted by "labelme" tool.

    input: json file annoteted by "labelme" with linestrip mode.
    ouput: polygon dict data converted from linestrip json file.
    """
    with open(line_json_path) as f:
        line_strip = json.load(f)

    copy_dict = copy.deepcopy(line_strip)

    X_MAX = copy_dict['imageWidth']
    Y_MAX = copy_dict['imageHeight']

    left_dict = copy_dict['shapes'][0]
    right_dict = copy_dict['shapes'][1]

    if include_front:
        # 画像手前をセグメンテーションに入れる場合
        # 座標は, 画像左上を原点とし, 右方向がx正, 下方向がy正
        # 出力用辞書

        left_first_point = left_dict["points"][-1]  # 一番手前の点
        left_second_point = left_dict["points"][-2]  # 二番目に手前の点

        # 大抵の場合負の値になる, yの増加量/xの増加量
        left_slope = (left_first_point[1] - left_second_point[1]) \
            / (left_first_point[0] - left_second_point[0])

        # 直線の延長が先に画像の左端に当たった場合
        if (0 - left_first_point[0])*left_slope + left_first_point[1] < Y_MAX:
            left_added_point = [[0, (0 - left_first_point[0])*left_slope + left_first_point[1]],
            [0, Y_MAX]]
        # 直線の延長が先に画像の下端に当たった場合
        else:
            left_added_point = [[left_first_point[0] + (Y_MAX - left_first_point[1])/left_slope ,Y_MAX]]

        right_first_point = right_dict["points"][-1]  # 一番手前の点
        right_second_point = right_dict["points"][-2]  # 二番目に手前の点

        # 大抵の場合正の値になる, yの増加量/xの増加量
        right_slope = (right_first_point[1] - right_second_point[1]) \
            / (right_first_point[0] - right_second_point[0])

        # 直線の延長が先に画像の右端に当たった場合
        if (X_MAX - right_first_point[0])*right_slope + right_first_point[1] < Y_MAX:
            right_added_point = [[X_MAX, (X_MAX - right_first_point[0])*right_slope + right_first_point[1]],
            [X_MAX, Y_MAX]]
        # 直線の延長が先に画像の下端に当たった場合
        else:
            right_added_point = [[right_first_point[0] + (Y_MAX - right_first_point[1])/right_slope ,Y_MAX]]

        left_dict['points'].extend(left_added_point)
        right_dict['points'].extend(right_added_point)
        right_dict['points'].reverse()
            
    else:
        # 画像手前をセグメンテーションに入れない場合
        right_dict['points'].reverse()  # 右ラインを逆順にしないとpolygonにならない

    left_dict["points"].extend(right_dict["points"])  # ラインの座標を連結

    # 左ラインだった場所をpolygon情報に上書き
    copy_dict['shapes'][0]['label'] = "road"
    copy_dict['shapes'][0]['points'] = left_dict['points']
    copy_dict['shapes'][0]['shape_type'] = 'polygon'

    # 右ライン情報を削除
    del copy_dict['shapes'][1]
    
    return copy_dict


def get_target_list(target_dir, pattern="*.json")->list:
    """
    探索対象ディレクトリからjsonファイルのパスのリストを作成
    """
    target_list = glob.glob(os.path.join(target_dir, pattern))
    return target_list


def save_json(save_dict, file_path:str):
    with open(file_path, "w") as f:
        json.dump(save_dict, f)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
    help='変換したいjsonファイルがあるディレクトリ')
    parser.add_argument('--output_dir', help=' 出力したいディレクトリ')
    args = parser.parse_args()
    os.mkdir(args.output_dir)
    target_list = get_target_list(args.input_dir)

    # 対象のjsonパスへの処理
    count = 0
    for target_path in target_list:
        base_name = target_path.split("/")[-1]  # ファイル名の取得

        if base_name.find("converted") == -1:
            count += 1
            out_name = "converted_" + base_name
            out_path = str(os.path.join(args.output_dir, out_name))
            polygon_dict = line2polygon(target_path, include_front=True)
            save_json(polygon_dict, out_path)
    print(f"{count} files are converted to {args.output_dir}")
