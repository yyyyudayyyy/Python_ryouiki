from ultralytics import YOLO
import cv2
import torch

# モデル読み込み
model = YOLO("yolo11x-pose.pt")

# 動画読み込み
cap1 = cv2.VideoCapture("ex3a.mp4")
cap2 = cv2.VideoCapture("ex3b.mp4")

if not cap1.isOpened() or not cap2.isOpened():
    print("動画を開けませんでした")
    exit()

# 骨格リンク
links = [
    [5, 7], [6, 8], [7, 9], [8, 10],
    [11, 13], [12, 14], [13, 15], [14, 16],
    [5, 11], [6, 12], [5, 6], [11, 12],
]

ignore_indices = [0, 1, 2, 3, 4]  # 顔のキーポイント除外

def draw_pose_on_gray(frame, results, color=(0,0,255)):
    draw = frame.copy()
    draw[:] = (128, 128, 128)  # 背景灰色
    if results is not None and results[0].keypoints is not None:
        all_people_nodes = results[0].keypoints.xy
        for nodes in all_people_nodes:
            for n1, n2 in links:
                if nodes[n1][0]*nodes[n1][1]*nodes[n2][0]*nodes[n2][1]==0:
                    continue
                cv2.line(draw,
                         nodes[n1].to(torch.int).tolist(),
                         nodes[n2].to(torch.int).tolist(),
                         color, 2)
            for i in range(len(nodes)):
                if i in ignore_indices:
                    continue
                x, y = nodes[i]
                if x*y==0:
                    continue
                cv2.circle(draw, (int(x.item()), int(y.item())), 4, color, -1)
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

    # 骨格描画（赤色に統一）
    draw_long = draw_pose_on_gray(frame_long, results_long, color=(0,0,255))
    if not short_finished:
        draw_short = draw_pose_on_gray(frame_short, results_short, color=(0,0,255))
        combined = cv2.addWeighted(draw_long, 0.5, draw_short, 0.5, 0)
    else:
        combined = draw_long  # 短い方終了後は長い方のみ描画

    cv2.imshow("Overlay Pose Detection", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
