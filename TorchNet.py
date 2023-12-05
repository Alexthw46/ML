import torch as t


class SimpleNN(t.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the layers
        self.fc1 = t.nn.Linear(self.input_size, self.hidden_size)  # Adjust the input size here
        self.linear_chain = t.nn.Linear(self.hidden_size, self.hidden_size)
        self.final = t.nn.Linear(self.hidden_size, self.output_size)

        self.bn1 = t.nn.BatchNorm1d(self.hidden_size)
        # Define the activation functions
        self.relu = t.nn.ReLU()
        self.sigmoid = t.nn.Sigmoid()
        self.tanh = t.nn.Tanh()

    def forward(self, x):
        # Define the forward pass
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.final(x)
        x = self.sigmoid(x)
        return x
