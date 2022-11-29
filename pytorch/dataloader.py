from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from lmdbdataset import lmdbdataset
import numpy as np
from utils import generate_pair_for_pytorch
class lmdb_dataset(Dataset):
    def __init__(self, lmdb_dir, shape,config):
        self.lmdb = lmdbdataset(lmdb_dir, shape,lock=False)
        self.key_list = self.lmdb.key_list()
        self.config = config
    def __getitem__(self, index):
        img = self.lmdb.read_image(self.key_list[index])
        img = img[None,:,:,:]
        x,y = generate_pair_for_pytorch(img, self.config)
        return x[0].copy(),y[0].copy()

    def __len__(self):
        return len(self.key_list)

    def __del__(self):
        self.lmdb.close()


if __name__ == "__main__":
    print("hello")
    from config import models_genesis_config

    conf = models_genesis_config()
    dataset = lmdb_dataset("/dataset/lmdb/debug3.lmdb",(64,64,32),conf)
    dataloader = DataLoader(dataset,32,True,num_workers=8,drop_last=True)
    for i in iter(dataloader):
        print(np.shape(i[0]))
        print(np.shape(i[1]))
