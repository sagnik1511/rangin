from PIL import Image
from glob import glob
from torchvision.transforms import ToTensor


class RDataset:

    def __init__(self, root, transform=None, size=(224, 224)):
        self.root = root
        self.size = size
        self.transform = transform if transform else ToTensor()
        self.files = sorted(glob(root + '/*.jpg'))

    def __len__(self):
        return len(self.files)

    def load_coloured(self, path):
        img = Image.open(path)
        img = img.resize(self.size)
        return img
    
    def load_bw(self, path):
        img = Image.open(path).convert('L')
        img = img.resize(self.size)
        return img

    def __getitem__(self, index):
        path = self.files[index]
        real = self.load_coloured(path)
        bw = self.load_bw(path)
        if self.transform is not None:
            real = self.transform(real)
            bw = self.transform(bw)
        return bw, real
