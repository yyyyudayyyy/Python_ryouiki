from ultralytics import YOLO
import cv2
import torch

model = YOLO("yolo11x-pose.pt")

results = model("ex1.jpg")
# 1人目のキー・ポイント(x,y)を獲得する．
# 2人目はresults[0].keypoints.data[1]に格納されている．
nodes = results[0].keypoints.data[0][:, :2]

# 骨格のリンク
links = [
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [11, 13],
    [12, 14],
    [13, 15],
    [14, 16],
    [5, 11],
    [6, 12],
    [5, 6],
    [11, 12],
    [4, 2],
    [2, 0],
    [0, 1],
    [1, 3],
    [4, 6],
    [5, 3],
]

# 入力画像
img = results[0].orig_img

for n1, n2 in links:
    # 誤認識のリンクを描画しない．
    if nodes[n1][0] * nodes[n1][1] * nodes[n2][0] * nodes[n2][1] == 0:
        continue
    cv2.line(
        img,
        # 2つの座標を整数化し，テンソルからリストにする．
        nodes[n1].to(torch.int).tolist(),
        nodes[n2].to(torch.int).tolist(),
        (0, 255, 0),
        thickness=2,
    )

# 画像を画面に表示する．
cv2.imshow("", img)
# キー入力があるまで，停止する．
cv2.waitKey(0)
# フレームを閉じる．
cv2.destroyAllWindows()