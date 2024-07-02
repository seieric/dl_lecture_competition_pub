import torch
import torchvision
import tqdm
import os
import re

# ①Imagesファイルへのパスを読み込む
# ②順番に画像を読み込み、画像をリサイズする
# ③リサイズした画像を番号を名前にしてファイルに保存する


def main():
    data_dir = "data"
    images_dir = "data/Images"
    images_out_dir = "data/preprocessed_images"
    for split in ["train", "val"]:
        print(f"Preprocessing {split} images...")
        os.makedirs(os.path.join(images_out_dir, split), exist_ok=True)
        with open(f"{data_dir}/{split}_image_paths.txt") as f:
            for i, line in enumerate(f):
                if i % 1000 == 0:
                    print(f"Processing image {i}")
                image_path = line.strip()
                # image_pathが存在しないときは与えられたデータがおかしいのでカテゴリのディレクトリ名を追加
                if not os.path.exists(os.path.join(images_dir, image_path)):
                    category_name = re.search(r"(.*)_[0-9]+.*", image_path).group(1)
                    image_path = os.path.join(category_name, image_path)
                # 画像を読み込む
                image = torchvision.io.read_image(os.path.join(images_dir, image_path))
                # 画像をリサイズ
                image = torchvision.transforms.Resize((224, 224))(image)
                # 画像を保存
                torchvision.io.write_png(
                    image, os.path.join(images_out_dir, split, f"{i}.png")
                )


if __name__ == "__main__":
    main()
