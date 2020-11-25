import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as data
from datasets import MyDataset


class CNN(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, hidden_dim,
                 kernel_num, kernel_dim, pooling_style="max", dp_rate=0):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.kernel_num = kernel_num
        self.kernel_dim = kernel_dim
        self.dp_rate = dp_rate
        self.conv_dim = self.input_dim - self.kernel_dim + 1
        if pooling_style == "mean":
            self.pooling = nn.AvgPool2d(2, 2)
        else:
            self.pooling = nn.MaxPool2d(2, 2)
        self.pool_dim = (self.conv_dim - 2) // 2 + 1

        self.conv = nn.Conv2d(1, self.kernel_num, self.kernel_dim)
        tmp_dim = 14400
        self.fc1 = nn.Linear(tmp_dim, self.hidden_dim)
        self.dp = nn.Dropout(self.dp_rate)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, img):
        x = torch.unsqueeze(img, 1)  # batch * 1 * 32 * 32
        conv_out = F.relu(self.conv(x))  # batch * kernel_num * conv_out * conv_out
        pool_out = self.pooling(conv_out).view(conv_out.shape[0], -1)  # batch * kernel_num * pool_out * pool_out
        fc1_out = torch.sigmoid(self.fc1(pool_out.unsqueeze(1)))
        fc2_out = self.fc2(self.dp(fc1_out))
        output = F.softmax(fc2_out.squeeze(), dim=1)
        return output


def train_cnn_pytorch():
    image_dim = 32
    hidden_dim = 200
    output_dim = 10
    kernel_dim = 3
    kernel_num = 64
    batch_size = 8
    lr = 0.01
    dp_rate = 0.3
    epochs = 200

    best_result = [0, 0]
    no_update = 0

    # os.environ["CUDA_VISIBLE_DEVICES"] = 0
    print("Start training")
    model = CNN(batch_size=batch_size, input_dim=image_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                kernel_num=kernel_num, kernel_dim=kernel_dim, dp_rate=dp_rate)
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_data = MyDataset("digits/trainingDigits")
        train_loader = data.DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=True)
        model.train()
        start = time.time()
        print(f"Epoch {epoch} start ")
        avg_loss = 0
        count = 0
        for step, input_data in enumerate(train_loader):
            x = torch.clone(input_data[0]).float()
            target = torch.clone(input_data[1]).long()
            if torch.cuda.is_available():
                x = x.cuda()
                target = target.cuda()
            prediction = model(x)
            loss = F.cross_entropy(prediction, target.argmax(dim=1))
            avg_loss += loss.item()
            count += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss /= len(train_data)
        end = time.time()
        print(f"Epoch {epoch} done, Train average loss: {avg_loss}, costing time: {end - start}")

        if epoch % 20 == 0:
            accuracy, wrong_numbers = evaluate_cnn_pytorch(model, batch_size)
            if accuracy > best_result[0]:
                best_result[0] = accuracy
                best_result[1] = wrong_numbers
                no_update = 0
            else:
                no_update += 1
        if no_update >= 5:
            print("Best Accuracy on test data: " + str(best_result[0]) + "%")
            print(f"Best wrong_numbers: {best_result[1]}")
            exit()
    print("Best Accuracy on test data: " + str(best_result[0]) + "%")
    print(f"Best wrong_numbers: {best_result[1]}")


def evaluate_cnn_pytorch(model, batch_size):
    model.eval()
    test_data = MyDataset("digits/testDigits")
    test_loader = data.DataLoader(test_data, batch_size=batch_size, num_workers=0, shuffle=False)
    ok_predictions = 0
    for step, test in enumerate(test_loader):
        x = torch.clone(test[0]).float()
        target = torch.clone(test[1]).long()
        if torch.cuda.is_available():
            x = x.cuda()
            target = target.cuda()
        predictions = model(x)
        for i in range(len(predictions)):
            expected = torch.argmax(target[i])
            prediction = torch.argmax(predictions[i])
            if expected == prediction:
                ok_predictions += 1

    accuracy = round((ok_predictions / len(test_data)) * 100, 2)
    wrong_numbers = len(test_data) - ok_predictions
    print("Accuracy on test data: " + str(accuracy) + "%")
    print(f"wrong_numbers: {wrong_numbers}")

    return accuracy, wrong_numbers


if __name__ == '__main__':
    train_cnn_pytorch()
