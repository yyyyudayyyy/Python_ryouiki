from ultralytics import YOLO
import cv2
import torch

model = YOLO("yolo11x-pose.pt")

cap = cv2.VideoCapture("ex2.mp4")

if not cap.isOpened():
    print("動画ファイルを開けませんでした。")
    exit()

links = [
    [5, 7], [6, 8], [7, 9], [8, 10],
    [11, 13], [12, 14], [13, 15], [14, 16],
    [5, 11], [6, 12], [5, 6], [11, 12],
]

ignore_indices = [0, 1, 2, 3, 4]


while True:
    # 1フレーム読み込み
    ret, frame = cap.read()
    
    # 動画終了または読み込み失敗でループを抜ける
    if not ret:
        break

    # 推論実行
    results = model(frame, verbose=False)
    
    # キーポイント情報がある場合のみ処理を実行
    if results[0].keypoints is not None:
        # 画像内にいる全ての人物のキーポイントを取得
        all_people_nodes = results[0].keypoints.xy

        # 検出された人数分ループする
        for nodes in all_people_nodes:
            
            for n1, n2 in links:
                # 座標が(0,0)の場合は未検出なので描画しない
                if nodes[n1][0] * nodes[n1][1] * nodes[n2][0] * nodes[n2][1] == 0:
                    continue
                
                cv2.line(
                    frame,
                    nodes[n1].to(torch.int).tolist(),
                    nodes[n2].to(torch.int).tolist(),
                    (0, 0, 255), 
                    thickness=2,
                )

            for i in range(len(nodes)):
                
                if i in ignore_indices:
                    continue

                x, y = nodes[i]
                # 未検出(0,0)はスキップ
                if x * y == 0:
                    continue

                center = (int(x.item()), int(y.item()))
                cv2.circle(
                    frame,
                    center,
                    4,
                    (0, 255, 255),  
                    thickness=-1,
                )

    # 画面に表示
    cv2.imshow("Pose Estimation", frame)

    # キー入力待機 (1ms)。'q'が押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()