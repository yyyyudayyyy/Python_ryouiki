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

def draw_pose_on_gray(frame, results):
    """灰色背景に骨格のみ描画"""
    draw = frame.copy()
    draw[:] = (128, 128, 128)  # 背景灰色

    if results[0].keypoints is not None:
        all_people_nodes = results[0].keypoints.xy
        for nodes in all_people_nodes:
            # ボーン
            for n1, n2 in links:
                if nodes[n1][0] * nodes[n1][1] * nodes[n2][0] * nodes[n2][1] == 0:
                    continue
                cv2.line(
                    draw,
                    nodes[n1].to(torch.int).tolist(),
                    nodes[n2].to(torch.int).tolist(),
                    (0, 0, 255),
                    2
                )
            # キーポイント
            for i in range(len(nodes)):
                if i in ignore_indices:
                    continue
                x, y = nodes[i]
                if x * y == 0:
                    continue
                cv2.circle(
                    draw,
                    (int(x.item()), int(y.item())),
                    4,
                    (0, 255, 255),
                    -1
                )
    return draw

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # 推論
    results1 = model(frame1, verbose=False)
    results2 = model(frame2, verbose=False)

    # 骨格描画（灰色背景）
    pose1 = draw_pose_on_gray(frame1, results1)
    pose2 = draw_pose_on_gray(frame2, results2)

    # 重ねる（透過率0.5）
    combined = cv2.addWeighted(pose1, 0.5, pose2, 0.5, 0)

    cv2.imshow("Overlay Pose Detection", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
