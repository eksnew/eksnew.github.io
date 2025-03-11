import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self) -> None:
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=2)
        self.fc2 = nn.Linear(in_features=2, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # layer 1
        print('layer1 input:', x)
        out = self.fc1(x)
        print('layer1 output:', out)
        out = self.sigmoid(out)
        print('layer1 output with Sigmoid:', out)
        # layer 2
        out = self.fc2(out)
        print('layer2 output:', out)
        out = self.sigmoid(out)
        print('layer2 output with Sigmoid:', out)

        return out

    def print_params(self):
        for i, param in enumerate(self.parameters()):
            print(i, param.data)


if __name__ == '__main__':
    model = NeuralNet()
    print('-' * 10 + 'Before setting weights' + '-' * 10)
    model.print_params()

    print('-' * 10 + 'After setting weights' + '-' * 10)
    model.fc1.weight = nn.Parameter(torch.tensor([[0.2, 0.4], [0.6, 0.8]], dtype=torch.float))
    model.fc1.bias = nn.Parameter(torch.tensor([0.2, 0.2], dtype=torch.float))
    model.fc2.weight = nn.Parameter(torch.tensor([[0.2, 0.8]], dtype=torch.float))
    model.fc2.bias = nn.Parameter(torch.tensor([0.2], dtype=torch.float))
    model.print_params()

    # Define data and Hyper-parameters
    original_input = torch.tensor([1, 0.5], dtype=torch.float)
    label = torch.tensor([0.114514], dtype=torch.float)
    learning_rate = 1
    num_epochs = 5

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimiser = torch.optim.SGD(params=model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # First Forward pass
        print('-' * 10 + 'epoch ' + str(epoch + 1) + '-' * 10)
        outputs = model(original_input)
        loss = criterion(outputs, label)
        print('OUTPUT:', outputs)
        print('loss:', loss)

        # Backward and optimize
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        print(model.fc2.weight.grad)

        model.print_params()
