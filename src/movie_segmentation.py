import cv2
import sys
import torch
import torch.nn as nn
from torchvision import transforms as T
import time
import os 
import segmentation_models_pytorch as smp  # Segmentationのモデル
import albumentations as A  # Data Augmentation
import numpy as np

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

model = smp.Unet(
    'mobilenet_v2',
    encoder_weights='imagenet',
    classes=2, 
    activation=None,
    encoder_depth=5, 
    decoder_channels=[256, 128, 64, 32, 16]
    )
model.load_state_dict(torch.load(STATE_DICT_PATH))

cap = cv2.VideoCapture(MP4_PATH)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 水路マスクの生成
    start = time.time()  # 計算時間

    pred_mask = predict_image_mask(model, frame)

    end = time.time()
    fps = np.round(1/(end - start), 1)

    pred_mask = pred_mask.unsqueeze(0)
    pred_mask = pred_mask.numpy()
    zero_array = np.zeros(shape=(2, HEIGHT, WIDTH))
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
    
    cv2.imshow("OyoNet output" ,blend)
    key = cv2.waitKey(1)
    if key == 27:
        sys.exit(0)