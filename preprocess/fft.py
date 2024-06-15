import torch
import os

def main():
  device = 'cuda:0'
  data_dir = "./data"
  for split in ["train", "val", "test"]:
    # データをGPUに読み込む
    print(f"Loading {split} data...")
    X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
    print("Data loaded. Shape: ", X.shape)

    # パワースペクトルを求める
    X_reshaped = X.view(X.shape[0], -1)
    batch_size = 512
    batch_num = X.shape[0] // batch_size + 1
    power_spectrum_list = []
    print("Converting to power spectrum...")
    for i in range(batch_num):
      print("Batch No.", i)
      batch_X_reshaped = X_reshaped[i*batch_size:(i+1)*batch_size].to(device)
      fft_data = torch.fft.fft(batch_X_reshaped)
      batch_power_spectrum = torch.abs(fft_data) ** 2
      power_spectrum_list.append(batch_power_spectrum)
      del batch_X_reshaped, fft_data
      torch.cuda.empty_cache()
    X = torch.cat(power_spectrum_list, dim=0).view(X.shape)
    print("Convertion finished.")
    for batch_power_spectrum in power_spectrum_list:
      del batch_power_spectrum
    torch.cuda.empty_cache()

    print("Final shape:", X.shape)
    
    print("Writing data...")
    torch.save(X.to('cpu'), os.path.join(data_dir, f"{split}_X_fft.pt"))
    print("Data written.")
    
    print("Erasing variables...")
    del X
    torch.cuda.empty_cache()

if __name__ == "__main__":
  main()