import logging
from ultralytics import YOLO
import cv2
import torch
import numpy as np
from scipy.spatial.distance import cdist

# --- ログ設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 設定 ---
MODEL_PATH = "yolo11x-pose.pt"
VIDEO1_PATH = "ex3a.mp4"
VIDEO2_PATH = "ex3b.mp4"
OUTPUT_PATH = "output_dtw.mp4"  # 保存するファイル名

# 骨格リンク定義
LINKS = [
    [5, 7], [6, 8], [7, 9], [8, 10],
    [11, 13], [12, 14], [13, 15], [14, 16],
    [5, 11], [6, 12], [5, 6], [11, 12],
]
IGNORE_INDICES = [0, 1, 2, 3, 4]

# --- 関数定義 ---

def extract_keypoints(results):
    if results is None or results[0].keypoints is None:
        return None
    kpts = results[0].keypoints.xy.cpu().numpy()[0]
    if np.all(kpts == 0):
        return None
    return kpts

def get_pose_feature(kpts):
    if kpts is None:
        return None
    
    hip_left = kpts[11]
    hip_right = kpts[12]
    
    if np.all(hip_left == 0) or np.all(hip_right == 0):
        return None

    center_x = (hip_left[0] + hip_right[0]) / 2
    center_y = (hip_left[1] + hip_right[1]) / 2

    normalized_kpts = kpts[5:, :] - np.array([center_x, center_y])
    return normalized_kpts.flatten()

def compute_dtw_path(seq1, seq2):
    logger.info(f"DTW計算開始: Seq1({len(seq1)}) vs Seq2({len(seq2)})")
    
    valid_len = len(seq1[0]) if seq1[0] is not None else 0
    if valid_len == 0:
        for s in seq1:
            if s is not None: valid_len = len(s); break
        if valid_len == 0: valid_len = 24

    seq1_arr = np.array([s if s is not None else np.zeros(valid_len) for s in seq1])
    seq2_arr = np.array([s if s is not None else np.zeros(valid_len) for s in seq2])

    dist_matrix = cdist(seq1_arr, seq2_arr, metric='euclidean')

    n, m = dist_matrix.shape
    cost = np.zeros((n, m))
    
    cost[0, 0] = dist_matrix[0, 0]
    for i in range(1, n):
        cost[i, 0] = cost[i-1, 0] + dist_matrix[i, 0]
    for j in range(1, m):
        cost[0, j] = cost[0, j-1] + dist_matrix[0, j]

    for i in range(1, n):
        for j in range(1, m):
            cost[i, j] = dist_matrix[i, j] + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])

    path = []
    i, j = n - 1, m - 1
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            options = [cost[i-1, j], cost[i, j-1], cost[i-1, j-1]]
            idx = np.argmin(options)
            if idx == 0: i -= 1
            elif idx == 1: j -= 1
            else: i -= 1; j -= 1
        path.append((i, j))
    
    logger.info("DTW計算完了")
    return path[::-1]

def draw_pose_from_kpts(img_shape, kpts, color=(0,0,255), target_center=None):
    img_h, img_w = img_shape
    draw = np.full((img_h, img_w, 3), (128, 128, 128), dtype=np.uint8) 

    if target_center is None:
        target_center = (img_w // 2, img_h // 2)
    tx, ty = target_center

    if kpts is None:
        return draw

    hip_left = kpts[11]
    hip_right = kpts[12]
    
    if np.all(hip_left == 0) or np.all(hip_right == 0):
        return draw

    current_cx = (hip_left[0] + hip_right[0]) / 2
    current_cy = (hip_left[1] + hip_right[1]) / 2

    offset_x = tx - current_cx
    offset_y = ty - current_cy

    for n1, n2 in LINKS:
        pt1 = kpts[n1]
        pt2 = kpts[n2]
        if pt1[0]*pt1[1]*pt2[0]*pt2[1] == 0: continue
        
        p1_draw = (int(pt1[0] + offset_x), int(pt1[1] + offset_y))
        p2_draw = (int(pt2[0] + offset_x), int(pt2[1] + offset_y))
        cv2.line(draw, p1_draw, p2_draw, color, 2)

    for i in range(len(kpts)):
        if i in IGNORE_INDICES: continue
        x, y = kpts[i]
        if x*y == 0: continue
        cx_draw = int(x + offset_x)
        cy_draw = int(y + offset_y)
        cv2.circle(draw, (cx_draw, cy_draw), 4, (0, 255, 255), -1)

    return draw

# --- メイン処理 ---

logger.info(f"モデル読み込み開始: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    logger.info("モデル読み込み完了")
except Exception as e:
    logger.error(f"モデル読み込みエラー: {e}")
    exit()

cap1 = cv2.VideoCapture(VIDEO1_PATH)
cap2 = cv2.VideoCapture(VIDEO2_PATH)

if not cap1.isOpened() or not cap2.isOpened():
    logger.error("動画を開けませんでした")
    exit()

# 動画情報の取得
fps = cap1.get(cv2.CAP_PROP_FPS)  # 書き出し用FPSとして1つ目の動画のFPSを使用
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_pos = (width // 2, height // 2)

# --- ステップ1: 特徴抽出 ---
logger.info("=== Step 1: 動画解析と特徴抽出 ===")

frames1, frames2 = [], []
feats1, feats2 = [], []
kpts_list1, kpts_list2 = [], []

# Video 1
logger.info(f"動画1 解析中...")
count = 0
while True:
    ret, frame = cap1.read()
    if not ret: break
    frames1.append(frame)
    results = model(frame, verbose=False)
    kpts = extract_keypoints(results)
    kpts_list1.append(kpts)
    feats1.append(get_pose_feature(kpts))
    
    count += 1
    if count % 50 == 0: logger.info(f"  Video 1: {count} frames")

# Video 2
logger.info(f"動画2 解析中...")
count = 0
while True:
    ret, frame = cap2.read()
    if not ret: break
    frames2.append(frame)
    results = model(frame, verbose=False)
    kpts = extract_keypoints(results)
    kpts_list2.append(kpts)
    feats2.append(get_pose_feature(kpts))

    count += 1
    if count % 50 == 0: logger.info(f"  Video 2: {count} frames")

cap1.release()
cap2.release()

if len(feats1) == 0 or len(feats2) == 0:
    logger.error("特徴量抽出失敗")
    exit()

# None補完
valid_feat_len = 0
for f in feats1:
    if f is not None: valid_feat_len = len(f); break
if valid_feat_len == 0: valid_feat_len = 24

dummy_feat = np.zeros(valid_feat_len)
proc_feats1 = [f if f is not None else dummy_feat for f in feats1]
proc_feats2 = [f if f is not None else dummy_feat for f in feats2]

# --- ステップ2: DTW計算 ---
logger.info("=== Step 2: DTW計算 ===")
path = compute_dtw_path(proc_feats1, proc_feats2)

# --- ステップ3: 動画保存と表示 ---
logger.info(f"=== Step 3: 動画生成と保存 ({OUTPUT_PATH}) ===")

# VideoWriterの初期化
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4形式
video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

cv2.namedWindow("DTW Aligned Pose Overlay", cv2.WINDOW_NORMAL)

frame_idx = 0
logger.info("保存開始...")

for idx1, idx2 in path:
    frame_idx += 1
    
    # 描画
    draw1 = draw_pose_from_kpts((height, width), kpts_list1[idx1], color=(255, 0, 0), target_center=center_pos)
    draw2 = draw_pose_from_kpts((height, width), kpts_list2[idx2], color=(0, 0, 255), target_center=center_pos)

    # 合成
    combined = cv2.addWeighted(draw1, 0.5, draw2, 0.5, 0)
    
    # 情報を書き込む（保存される動画にも文字が入ります）
    info_text = f"Path: {frame_idx}/{len(path)} | V1:{idx1} V2:{idx2}"
    cv2.putText(combined, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 動画ファイルに書き込み
    video_writer.write(combined)

    # 画面表示
    cv2.imshow("DTW Aligned Pose Overlay", combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("中断されました")
        break

# 後処理
video_writer.release() # 重要: これを忘れると動画が壊れます
cv2.destroyAllWindows()
logger.info(f"完了: {OUTPUT_PATH} に保存しました")