from ultralytics import YOLO
import cv2
import torch
import numpy as np

# モデル読み込み
model = YOLO("yolo11x-pose.pt")

# 動画読み込み
cap1 = cv2.VideoCapture("ex3a.mp4")
cap2 = cv2.VideoCapture("ex3b.mp4")

if not cap1.isOpened() or not cap2.isOpened():
    print("動画を開けませんでした")
    exit()

# 骨格リンク定義
links = [
    [5, 7], [6, 8], [7, 9], [8, 10],
    [11, 13], [12, 14], [13, 15], [14, 16],
    [5, 11], [6, 12], [5, 6], [11, 12],
]

ignore_indices = [0, 1, 2, 3, 4]  # 顔のキーポイント除外

def draw_aligned_pose(frame, results, color=(0,0,255), target_center=None):
    
    img_h, img_w = frame.shape[:2]
    draw = np.full((img_h, img_w, 3), (128, 128, 128), dtype=np.uint8) 

    if target_center is None:
        target_center = (img_w // 2, img_h // 2)
    
    tx, ty = target_center

    if results is not None and results[0].keypoints is not None:
        
        all_people_nodes = results[0].keypoints.xy.cpu().numpy()
        
        for nodes in all_people_nodes:
            hip_left = nodes[11]
            hip_right = nodes[12]
            
            if np.all(hip_left == 0) or np.all(hip_right == 0):
                continue

            current_cx = (hip_left[0] + hip_right[0]) / 2
            current_cy = (hip_left[1] + hip_right[1]) / 2

            offset_x = tx - current_cx
            offset_y = ty - current_cy

            for n1, n2 in links:
                pt1 = nodes[n1]
                pt2 = nodes[n2]
                
                # 未検出の点は描画しない
                if pt1[0]*pt1[1]*pt2[0]*pt2[1] == 0:
                    continue
                
                # 座標をシフトして整数型に変換
                p1_draw = (int(pt1[0] + offset_x), int(pt1[1] + offset_y))
                p2_draw = (int(pt2[0] + offset_x), int(pt2[1] + offset_y))
                
                cv2.line(draw, p1_draw, p2_draw, color, 2)

            # キーポイント（点）の描画
            for i in range(len(nodes)):
                if i in ignore_indices:
                    continue
                x, y = nodes[i]
                if x*y == 0:
                    continue
                
                # 座標をシフト
                cx_draw = int(x + offset_x)
                cy_draw = int(y + offset_y)
                
                cv2.circle(draw, (cx_draw, cy_draw), 4, (0, 255, 255), -1)

    return draw

# 動画の長さを比較して長い方を決める
length1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
length2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

if length1 >= length2:
    long_cap = cap1
    short_cap = cap2
else:
    long_cap = cap2
    short_cap = cap1

short_finished = False

width = int(long_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(long_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_pos = (width // 2, height // 2)

while True:
    # 短い方が終了していない場合
    if not short_finished:
        ret_short, frame_short = short_cap.read()
        if not ret_short:
            short_finished = True
            frame_short = None
            results_short = None
        else:
            results_short = model(frame_short, verbose=False)
    else:
        frame_short = None
        results_short = None

    # 長い方は最後まで読み込む
    ret_long, frame_long = long_cap.read()
    if not ret_long:
        break
    results_long = model(frame_long, verbose=False)

    draw_long = draw_aligned_pose(frame_long, results_long, color=(255, 0, 0), target_center=center_pos)
    
    if not short_finished:
        draw_short = draw_aligned_pose(frame_short, results_short, color=(0, 0, 255), target_center=center_pos)
       
        combined = cv2.addWeighted(draw_long, 0.5, draw_short, 0.5, 0)
    else:
        combined = draw_long  # 短い方終了後は長い方のみ描画

    cv2.imshow("Overlay Aligned Pose", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()