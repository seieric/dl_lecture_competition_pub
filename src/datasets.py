import os
import torch
import torchvision


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", with_image=False) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.data_dir = data_dir
        self.with_image = with_image

        # 脳波データ
        data_path = os.path.join(data_dir, f"{split}_X_preprocessed.pt")
        if not os.path.exists(data_path):
            raise Exception("下処理されたデータが存在しません．")
        self.X = torch.load(data_path)
        # 被験者情報
        self.subject_idxs = torch.load(
            os.path.join(data_dir, f"{split}_subject_idxs.pt")
        )

        if split in ["train", "val"]:
            # 画像のクラス
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert (
                len(torch.unique(self.y)) == self.num_classes
            ), "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            if not self.with_image:
                return self.X[i], self.y[i], self.subject_idxs[i]
            image = torchvision.io.read_image(
                os.path.join(
                    self.data_dir, "preprocessed_images", self.split, f"{i}.png"
                )
            )
            image = image.float() / 255.0
            return self.X[i], self.y[i], self.subject_idxs[i], image
        else:
            return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
