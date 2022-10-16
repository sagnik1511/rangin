import os
import argparse
from tqdm import tqdm


def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir", required=True,
                        metavar="",
                        help="Directory containing images")

    return parser


def main(data_dir):
    for index, img in tqdm(enumerate(os.listdir(data_dir))):
        src_filepath = os.path.join(data_dir, img)
        dst_filepath = os.path.join(data_dir,
                                    str(index) + '.' + img.split(".")[-1])
        os.rename(src_filepath, dst_filepath)
    print("processed  finished...")


if __name__ == '__main__':
    parser = generate_parser()
    args = parser.parse_args()
    main(args.data_dir)
