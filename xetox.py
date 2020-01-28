#PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
#Common
import argparse
import pandas as pd
import codecs

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_num):
        self.data_num = data_num
        #データの前処理が必要、、、
        with codecs.open("./data/train_data.csv", "r", "Shift-JIS", "ignore") as file:
            self.df = pd.read_csv(file, delimiter=",", names=["年","月","日","馬名","馬番","枠番","年齢","性別","馬体重","斤量","場所","頭数","距離","馬場状態","天候","人気","単勝オッズ","確定着順","タイムS","着差タイム","トラックコード"])
            #print(df)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = torch.tensor(self.df[["馬番","枠番","年齢","馬体重","斤量"]].iloc[idx])
        out_label = torch.tensor([float(self.df["確定着順"].iloc[idx])])

        return out_data, out_label

def train(args, model, device, train_loader, optimizer, epoch):
    criterion = nn.MSELoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(batch_idx)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of horse racing prediction')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    data_set = MyDataset(16)
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=4, shuffle=False)

    # for i in train_loader:
    #     print(i)

    with codecs.open("./data/train_data.csv", "r", "Shift-JIS", "ignore") as file:
        df = pd.read_csv(file, delimiter=",", names=["年","月","日","馬名","馬番","枠番","年齢","性別","馬体重","斤量","場所","頭数","距離","馬場状態","天候","人気","単勝オッズ","確定着順","タイムS","着差タイム","トラックコード"])
        print(df)
    # print(df.iloc[2])
    # #torch_tensor = torch.tensor(df["馬体重"].values)
    # torch_tensor = torch.tensor(df["確定着順"].iloc[2])
    # print(torch_tensor)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(torch.cuda.is_available())

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        #test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == '__main__':
    main()