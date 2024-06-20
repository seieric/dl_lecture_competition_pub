# センサーの位置をモデルに組み込むためにlayoutを取得し、0-1に変換するスクリプト
import requests
import csv
from io import StringIO
import numpy as np
import torch
import os

# 使わないチャンネル
channels_to_skip = [
    "MLF25",
    "MRF43",
    "MRO13",
    "MZP01",
    "COMNT",
    "SCALE"
]

def main():
    response = requests.get("https://raw.githubusercontent.com/fieldtrip/fieldtrip/master/template/layout/CTF275.lay")

    if response.status_code != 200:
        print("取得失敗", response.status_code)
        exit()

    csv_data = response.text
    csv_io = StringIO(csv_data)
    reader = csv.reader(csv_io, delimiter=' ')

    coordinates = []

    for row in reader:
        if row[5] not in channels_to_skip:
            x = float(row[1])
            y = float(row[2])
            coordinates.append([x, y])
    coordinates = np.array(coordinates)

    mins = np.min(coordinates, axis=0)
    maxs = np.max(coordinates, axis=0)
    coordinates = (coordinates - mins) / (maxs - mins)   

    coordinates = torch.from_numpy(coordinates)
    torch.save(coordinates, os.path.join("./data", "layout.pt"))    

if __name__ == "__main__":
    main()