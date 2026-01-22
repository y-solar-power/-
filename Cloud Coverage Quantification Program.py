import cv2
import numpy as np
import glob
import os
import shutil
import matplotlib.pyplot as plt
import subprocess
from datetime import datetime

# ===== ffmpeg パス =====
ffmpeg_path = r"E:\ffmpeg-2025-12-22-git-c50e5c7778-full_build\ffmpeg-2025-12-22-git-c50e5c7778-full_build\bin\ffmpeg.exe" #ffpeg.exeを呼び出して

# ===== 一時フレーム出力フォルダ（絶対パス）=====
script_dir = os.path.dirname(os.path.abspath(__file__))
temp_folder_path = os.path.join(script_dir, "cloud_movie_gen")


# ===== 雲量計算関数 =====
def calculate_cloud_metrics(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None, None, None, None

    image = cv2.resize(image, (300, 300))
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([80, 50, 180]))
    mask2 = cv2.inRange(hsv, np.array([160, 0, 0]), np.array([180, 50, 180]))
    maskCloud = cv2.bitwise_or(mask1, mask2)

    maskedCloud = cv2.bitwise_and(image, image, mask=maskCloud)

    cloud_pixels = np.count_nonzero(maskCloud)
    cloud_rate = cloud_pixels / maskCloud.size * 100

    if cloud_pixels > 0:
        cloud_mask = maskCloud == 255
        cloud_v = hsv[cloud_mask, 2]
        cti_hsv = np.mean(cloud_v) / 255 * 100

        cloud_rgb = image_rgb[cloud_mask].astype(np.float32)
        rgb_bright = np.mean(np.mean(cloud_rgb, axis=1))
        cti_rgb_bright = rgb_bright / 255 * 100

        rgb_std = np.std(cloud_rgb, axis=1)
        cti_rgb_white = max(0, (120 - np.mean(rgb_std)) / 120) * 100

        cti = (cti_hsv + cti_rgb_bright + cti_rgb_white)/3
    else:
        cti = 0.0

    eco = 0.7 * cloud_rate + 0.3 * cti

    return cloud_rate, cti, eco, image, maskCloud, maskedCloud


# ===== 親フォルダ入力 =====
root_dir = input("親フォルダのパス: ").strip()

date_dirs = [
    os.path.join(root_dir, d)
    for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
]

for date_dir in date_dirs:
    print(f"\n=== Processing {date_dir} ===")

    # 日付フォルダ直下の画像フォルダ（例: 20241212）
    subdirs = [
        os.path.join(date_dir, d)
        for d in os.listdir(date_dir)
        if os.path.isdir(os.path.join(date_dir, d))
    ]

    if not subdirs:
        print("  画像フォルダなし → スキップ")
        continue

    image_dir = subdirs[0]       # 入力画像フォルダ
    parent_dir = date_dir        # 出力先

    images = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.JPG"))
    )

    if not images:
        print("  JPGなし → スキップ")
        continue

    # 一時フォルダ作成
    if os.path.exists(temp_folder_path):
        shutil.rmtree(temp_folder_path)
    os.makedirs(temp_folder_path, exist_ok=True)

    # 出力ファイル
    output_txt = os.path.join(parent_dir, "value_rgb111.txt")
    output_movie = os.path.join(parent_dir, "雲量_ECO_RGB111.mp4")

    f = open(output_txt, "w", encoding="utf-8")
    f.write("CloudRate(%), CTI(%), ECO(%)\n")

    cloud_rate_list, cti_list, eco_list = [], [], []
    fig, ax = plt.subplots()

    for i, image_path in enumerate(images):
        result = calculate_cloud_metrics(image_path)
        if result[0] is None:
            continue

        cloud_rate, cti, eco, image, maskCloud, maskedCloud = result

        cloud_rate_list.append(cloud_rate)
        cti_list.append(cti)
        eco_list.append(eco)

        ax.clear()
        ax.plot(cloud_rate_list, label="Cloud Rate")
        ax.plot(cti_list, linestyle=":", label="CTI")
        ax.plot(eco_list, linestyle="--", label="ECO")
        ax.set_ylim(0, 100)
        ax.legend(fontsize="small")
        fig.canvas.draw()

        graph_img = np.array(fig.canvas.renderer.buffer_rgba())
        graph_img = cv2.cvtColor(graph_img, cv2.COLOR_RGBA2BGR)

        outside = cv2.rotate(cv2.imread(image_path), cv2.ROTATE_90_CLOCKWISE)

        imgs = [
            cv2.resize(graph_img, (400, 400)),
            cv2.resize(outside, (400, 400)),
            cv2.resize(maskedCloud, (400, 400))
        ]

        combined = cv2.hconcat(imgs)

        basename = os.path.splitext(os.path.basename(image_path))[0]
        try:
            dt = datetime.strptime(basename, "%Y%m%d%H%M%S")
            date_str = dt.strftime("%Y/%m/%d %H:%M:%S")
        except:
            date_str = basename

        cv2.putText(
            combined, date_str, (10, 30),
            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2
        )

        cv2.imwrite(os.path.join(temp_folder_path, f"{i:05d}.jpg"), combined)
        f.write(f"{cloud_rate:.2f}, {cti:.2f}, {eco:.2f}\n")

    plt.close()
    f.close()

    # ===== 動画生成 =====(出力確認用)
    subprocess.run([
    ffmpeg_path, "-y", "-r", "60",
    "-i", os.path.join(temp_folder_path, "%05d.jpg"),
    "-vcodec", "libx264", "-pix_fmt", "yuv420p",
    "-r", "60", output_movie
    ], check=True, shell=True)
    
    shutil.rmtree(temp_folder_path)
    print(f"  完了 → {output_movie}")

print("\n=== 全フォルダ処理完了 ===")

