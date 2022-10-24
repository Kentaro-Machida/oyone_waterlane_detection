import cv2
import sys
import torch
from torchvision import transforms as T
import time
import segmentation_models_pytorch as smp  # Segmentationのモデル
import numpy as np
import argparse

HEIGHT = 320
WIDTH = 480
MP4_PATH = '../raw_data/9_6/2022_9_3_16_54/trimed.mp4'
STATE_DICT_PATH = '../trained_models/400_dataset/Unet-Mobilenet_v2_mIoU-0.939.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=2):
    """
    IoUの計算関数
    pred_mask: モデルが出力した画素ごとのクラスごとの確率
    mask: 正解mask(int64のtensor)
    """
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)  # (N, C, H, W)のCにsoftmax
        pred_mask = torch.argmax(pred_mask, dim=1)  # 画素ごとの最大クラスを取り出す
        pred_mask = pred_mask.contiguous().view(-1)  # maskを要素順にメモリに並べ(contiguous)1次元に(view)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes):  # loop per pixel class
            # clas ラベルの部分を1としたマスクを作成
            true_class = pred_mask == clas  
            true_label = mask == clas

            # 正解画像中に指定クラスのピクセルが1つもなければ, iou=0とする
            if true_label.long().sum().item() == 0:  # no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                # 0以外の画素をTrue, 0をFalseとしてandとorをとる
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union + smooth)
                iou_per_class.append(iou)
        
        # nanを無視したIoUの平均値mIoUを返す
        return np.nanmean(iou_per_class)

def predict_image_mask(model, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device); image=image.to(device)
    with torch.no_grad():
        
        image = image.unsqueeze(0)  # 3D tensor -> 4D tensor
        
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp4_path',help='入力したいmp4動画のパス')
    parser.add_argument('-save' ,'--save_path',help='保存したいmp4動画のパス。指定なしなら保存なし')
    parser.add_argument('--state_dict', help='学習済みモデルへのパス')
    parser.add_argument('--width', help='リサイズ横幅', default=480)
    parser.add_argument('--height', help='リサイズ縦幅', default=320)
    args = parser.parse_args()

    model = smp.Unet(
    'mobilenet_v2',
    encoder_weights='imagenet',
    classes=2, 
    activation=None,
    encoder_depth=5, 
    decoder_channels=[256, 128, 64, 32, 16]
    )
    model.load_state_dict(torch.load(args.state_dict))

    cap = cv2.VideoCapture(args.mp4_path)
    if type(args.save_path)==str:
        # コマンドラインで出力パスが指定されていれば, writter準備
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        writer = cv2.VideoWriter(args.save_path, fmt, 6, (args.width, args.height))
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (args.width, args.height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 水路マスクの生成
        start = time.time()  # 計算時間

        pred_mask = predict_image_mask(model, frame)

        end = time.time()
        fps = np.round(1/(end - start), 1)

        pred_mask = pred_mask.unsqueeze(0)
        pred_mask = pred_mask.numpy()

        # 船の行き先を決定
        # ある一定の高さでで切断して台形を作成
        # 台形の上底の中点を目的点とする
        hist = np.sum(pred_mask[0, int(args.height/3):, :], axis=0)
        dst_arrow = (int(np.mean(np.where(hist==hist.max()))), int(args.height/2))

        zero_array = np.zeros(shape=(2, args.height, args.width))
        pred_mask = np.concatenate((pred_mask, zero_array), axis=0)
        pred_mask = pred_mask.transpose(1, 2, 0).astype(np.uint8)*255

        # 表示用の画像生成
        blend = (pred_mask * 0.2 + frame*0.8).astype(np.uint8)
        blend = cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)
        cv2.putText(blend,"fps: " + str(fps),
        (20, 20),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        color=(255, 0, 0),
        fontScale=1.0,
        thickness=2
        )

        # 進行方向の描画
        cv2.arrowedLine(
            blend, pt1=(int(args.width/2), args.height), pt2=dst_arrow,
            color=(0, 0, 255), thickness=3
        )

        if type(args.save_path)==str:
            writer.write(blend)
        
        cv2.imshow("OyoNet output" ,blend)
        key = cv2.waitKey(1)
        if key == 27 or ret == False:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
