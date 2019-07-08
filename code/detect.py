import argparse

from data_loader import DataLoader
from plant_extractor import PlantExtractor
from utils import get_filename


def call_params():
    """ Parse the script parameters from bash input. """

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path",
                        type=str,
                        help="path to the folder containing the input images.")

    parser.add_argument("--save_path",
                        type=str,
                        help="path to the result folder.")

    parser.add_argument("-verbose",
                        action='store_false',
                        help="whether or not to allow for verbose")
    args = parser.parse_args()

    return args


def detect_plant(args):
    """ Apply the plant detection pipeline. """

    verboseprint = print if args.verbose else lambda *a, **k: None

    data_loader = DataLoader(args.data_path)
    plant_extractor = PlantExtractor(args.save_path)
    verboseprint("Objects have been instantiated.")

    dataset = data_loader.load_images()
    verboseprint("The dataset has been loaded.")

    for i in range(len(dataset)):
        input_name = get_filename(data_loader.images_path[i])
        new_name = input_name.split('.')[0] + '_with_bb.' + input_name.split('.')[-1]
        plant_extractor.save_img_with_bounding_box(dataset[i], new_name)


if __name__ == "__main__":
    args = call_params()
    detect_plant(args)