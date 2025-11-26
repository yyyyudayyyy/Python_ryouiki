from ultralytics import YOLO
import cv2
import torch

model = YOLO("yolo11x-pose.pt")

cap = cv2.VideoCapture("ex3a.mp4")

if not cap.isOpened():
    print("動画ファイルを開けませんでした。")
    exit()

# ボーンの接続
links = [
    [5, 7], [6, 8], [7, 9], [8, 10],
    [11, 13], [12, 14], [13, 15], [14, 16],
    [5, 11], [6, 12], [5, 6], [11, 12],
]

# 顔のキーポイントを除外
ignore_indices = [0, 1, 2, 3, 4]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- 背景を灰色で塗りつぶす（ここだけ変更ポイント） ----
    draw_frame = frame.copy()
    draw_frame[:, :] = (150, 150, 150)  # 完全な灰色で塗りつぶす

    # 推論
    results = model(frame, verbose=False)

    if results[0].keypoints is not None:
        all_people_nodes = results[0].keypoints.xy

        for nodes in all_people_nodes:

            # --- ボーン描画 ---
            for n1, n2 in links:
                if nodes[n1][0] * nodes[n1][1] * nodes[n2][0] * nodes[n2][1] == 0:
                    continue
                cv2.line(
                    draw_frame,
                    nodes[n1].to(torch.int).tolist(),
                    nodes[n2].to(torch.int).tolist(),
                    (0, 0, 255),  # 赤
                    thickness=2,
                )

            # --- キーポイント描画（顔以外） ---
            for i in range(len(nodes)):
                if i in ignore_indices:
                    continue

                x, y = nodes[i]
                if x * y == 0:
                    continue

                center = (int(x.item()), int(y.item()))
                cv2.circle(
                    draw_frame,
                    center,
                    4,
                    (0, 255, 255),  # 黄
                    thickness=-1,
                )

    # 表示
    cv2.imshow("Pose Estimation", draw_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
