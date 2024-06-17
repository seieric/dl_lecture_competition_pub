import torch
import os

def main():
  device = 'cuda:0'
  data_dir = "./data"
  for split in ["train", "val", "test"]:
    # データをGPUに読み込む
    print(f"Loading {split} data...")
    data_path = os.path.join(data_dir, f"{split}_X.pt")
    X = torch.load(data_path).to(device)
    print("Data loaded. Shape: ", X.shape)

    # チャンネルごとの最大値と最小値を求めるため、変形する
    X_reshaped = X.view(X.shape[1], X.shape[0]*X.shape[2])
    maxs, _ = torch.max(X_reshaped, dim=-1)
    mins, _ = torch.min(X_reshaped, dim=-1)
    divisor = torch.maximum(torch.abs(maxs), torch.abs(mins))
    # 各値を-1から1に収める
    print("Scaling...")
    X_processed = X_reshaped / divisor.view(-1,1)
    X_processed = X_processed.view(X.shape)

    print("Final shape:", X_processed.shape)
    
    print("Writing data...")
    torch.save(X_processed.to('cpu'), os.path.join(data_dir, f"{split}_X_preprocessed.pt"))
    print("Data written.")
    
    print("Erasing variables...")
    del X, X_reshaped, X_processed, divisor, _
    torch.cuda.empty_cache()

if __name__ == "__main__":
  main()