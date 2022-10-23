import os
import yaml
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage


def image_from_path(path):
    if not os.path.exists(path):
        print("File does not exist.")
        return FileNotFoundError
    image = Image.open(path)
    image = image.resize((224, 224))
    image = ToTensor()(image)
    return image


def load_model_config(path):
    if not os.path.exists(path):
        print("Config file does not exist.")
        raise FileNotFoundError
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
        f.close()
        name = cfg["name"]
        weights_path = cfg["weights"]
        del cfg["name"]
        del cfg["weights"]
        return name, cfg, weights_path


def save_outputs(data, output_folder):
    if data.device != "cpu":
        data = data.cpu()
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    transform = ToPILImage()
    for index in range(data.shape[0]):
        image = transform(data[index, ...])
        image.save(os.path.join(output_folder, f"{index}.png"))
    print(f"Results are stored in {output_folder}")
