import torch
import os

def main():
  device = 'cuda:0'
  data_dir = "./data"
  for split in ["train", "val", "test"]:
    # データをGPUに読み込む
    print(f"Loading {split} data...")
    X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).to(device)
    print("Data loaded. Shape: ", X.shape)
    # チャンネルごとの平均値と標準偏差を求めるため、変形する
    X_reshaped = torch.reshape(X, (X.shape[1], X.shape[0]*X.shape[2]))
    means = torch.mean(X_reshaped, dim=-1)
    stds = torch.std(X_reshaped, dim=-1)
    # 平均値から標準偏差3個分以上ずれている値は異常値とみなしてトリムする
    limits_top = means + stds*3
    limits_bottom = means - stds*3
    print("Clamping...")
    X = torch.clamp(X, min=limits_bottom.view(-1, 1), max=limits_top.view(-1, 1))
    # チャンネルごとの最大値と最小値を求めるため、変形する
    X_reshaped = torch.reshape(X, (X.shape[1], X.shape[0]*X.shape[2]))
    maxs, _ = torch.max(X_reshaped, dim=-1)
    mins, _ = torch.min(X_reshaped, dim=-1)
    divisor = torch.maximum(torch.abs(maxs), torch.abs(mins))
    # 各値を-1から1に収める
    print("Scaling...")
    X = X / divisor.view(-1,1)
    
    print("Final shape:", X.shape)
    
    print("Writing data...")
    torch.save(X.to('cpu'), os.path.join(data_dir, f"{split}_X_preprocessed.pt"))
    print("Data written.")
    
    print("Erasing variables...")
    del X, X_reshaped, means, stds, maxs, mins, divisor, limits_bottom, limits_top, _
    torch.cuda.empty_cache()

if __name__ == "__main__":
  main()