import os
import glob
import random
import codecs


root_path = '/home/hra/dataset/FLARE21/downsample/'
seed = 1997


if __name__ == '__main__':
    volume_path_list = glob.glob(root_path + '/img/*')
    volume_path_list = [os.path.basename(volume_path) for volume_path in volume_path_list]
    random.seed(seed)
    random.shuffle(volume_path_list)

    fold_num = 5

    fold_len = len(volume_path_list) // fold_num

    volume_fold_total = []
    for i in range(fold_num):
        if i != fold_num-1:
            single_fold = volume_path_list[i*fold_len:(i+1)*fold_len]
            volume_fold_total.append(single_fold)
        else:
            single_fold = volume_path_list[i*fold_len:]
            volume_fold_total.append(single_fold)

    for i in range(fold_num):
        fold_range = [j for j in range(fold_num)]
        fold_range.pop(i)
        test_list = volume_fold_total[i]
        train_list = []
        for j in fold_range:
            train_list += volume_fold_total[j]
        with codecs.open('../../data/fold_%d.txt' % i, mode='w', encoding='utf-8') as f:
            f.write('train:\n')
            for line in train_list:
                f.write(line + '\n')
            f.write('test:\n')
            for line in test_list:
                f.write(line + '\n')
        print('fold %d is done' % i)
        # print('d')