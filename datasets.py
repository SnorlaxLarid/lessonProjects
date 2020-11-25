import os
import random
import numpy as np
import torch.utils.data as data


class MyDataset(data.Dataset):
    def __init__(self, path):
        self.inputs, self.targets = load_datasets(path, dim=2)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)


def load_data(file, dim=1):
    with open(file, "r") as f:
        digit = [[int(dot) for dot in line[:-1]] for line in f.readlines()]
        digit = np.array(digit)
        if dim == 1:
            digit.resize((1, 1024))
        # plt.imshow(digit, cmap='gray')
        # plt.show()
    return digit


def load_datasets(path, dim=1):
    inputs = []
    targets = []
    file_list = os.listdir(path)
    shuffle_index = random.sample(range(len(file_list)), len(file_list))
    for i in shuffle_index:
        inputs.append(load_data(os.path.join(path, file_list[i]), dim=dim))
        index = int(file_list[i].split('/')[-1][0])
        targets.append([int(i == index) for i in range(10)])
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets

# load_data("digits/trainingDigits/0_0.txt")
