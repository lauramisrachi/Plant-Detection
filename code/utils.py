import os

EXT = [".png", ".jpg", ".jpeg", ".tiff"]


def get_images_filenames(folder):
    """ Returns the images path contained in the input folder """

    files_list = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(tuple(EXT)):
                files_list.append(os.path.join(root, file))

    return files_list


def get_filename(path):
    """ Returns the name of a file from its full path. """

    return path.split('/')[-1]
