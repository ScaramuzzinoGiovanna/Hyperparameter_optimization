from glob import glob
import os
from distutils.dir_util import copy_tree
import shutil
import argparse


def generate_db(name_folder, size_db):
    paths_img_0 = []
    paths_img_1 = []
    count0 = count1 = 0
    for person in glob(os.path.join(name_folder, '*')):
        folder0 = os.path.join(person, '0')
        folder1 = os.path.join(person, '1')
        for elem0 in glob(os.path.join(folder0, '*')):
            if count0 < size_db // 2:
                count0 = count0 + 1
                paths_img_0.append(elem0)
            else:
                break;

        for elem1 in glob(os.path.join(folder1, '*')):
            if count1 < size_db // 2:
                count1 = count1 + 1
                paths_img_1.append(elem1)
            else:
                break;
    return paths_img_0, paths_img_1


def split_train_test(path_imgs, path_train, path_test, dst, n_test):
    n_test_mezz = n_test // 2
    count = 0
    for elem in path_imgs:
        if count < n_test_mezz:
            count = count + 1
            shutil.copy(elem, os.path.join(path_test, dst))
        else:
            shutil.copy(elem, os.path.join(path_train, dst))


def createFolder(name, path_tmp):
    path = os.path.join(path_tmp, name)
    try:
        os.makedirs(path)
        print(
            "Succesfully created the directory {}".format(name))
    except OSError:
        print('creation of the directory {} failed'.format(name))
    return path


def copyImages(folder_images, dst, vect_id):
    for id in vect_id:
        src = os.path.join(folder_images, id)
        copy_tree(src, dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--sizeDB", required=False, default=60000, help="size new dataset")
    args = vars(ap.parse_args())
    perc_test = 20
    name_general_folder = 'db'
    path_images = os.path.join(name_general_folder, 'breast-histopathology-images')

    path_train = createFolder('train', os.path.join(name_general_folder, 'db_' + str(args['sizeDB'])))
    path_test = createFolder('test', os.path.join(name_general_folder, 'db_' + str(args['sizeDB'])))

    createFolder('0', path_train)
    createFolder('0', path_test)
    createFolder('1', path_train)
    createFolder('1', path_test)

    path0, path1 = generate_db(path_images, int(args['sizeDB']))

    n_test = (int(args['sizeDB']) * perc_test) // 100
    split_train_test(path0, path_train, path_test, '0', n_test)
    split_train_test(path1, path_train, path_test, '1', n_test)
