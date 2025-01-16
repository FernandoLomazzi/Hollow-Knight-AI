import torchvision.models as models
from torch import cuda, zeros, Size, nn
import torch

class ResnetModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lstm_dropout=0.0, bi=True):
        super(ResnetModel, self).__init__()
        self.resnet18 = models.resnet18(weights='DEFAULT').to("cuda" if cuda.is_available() else "cpu")
        self.resnet18.fc = nn.Identity()        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi = bi
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=lstm_dropout, batch_first=True, bidirectional=bi)
        
        if self.bi:
            hidden_size *= 2
          
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.resnet18(x)

        x = x.view(-1, 3, 512)
        
        num_layers = 2*self.num_layers if self.bi else self.num_layers
        batch_size, seq_len, feature_size = x.size()

        h0 = zeros(num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = zeros(num_layers, batch_size, self.hidden_size, device=x.device)

        x, (h, c) = self.rnn(x, (h0, c0))

        # x -> [batch_size, sequence_length, hidden_size]
        # h -> [num_layers, batch_size, hidden_size]
        # x[:, -1, :] == h[-1, :, :]
        x = self.mlp(x[:, -1, :])
        return x

    def encode(self, x):
        assert((3, 224, 224) == x.shape)
        x = self.resnet18(x.unsqueeze(0)).squeeze(dim=0)
        assert(Size([512]) == x.shape)
        return x

    def predict_from_encoding(self, x):
        assert((3, 512) == x.shape)
        num_layers = 2*self.num_layers if self.bi else self.num_layers
        seq_len, feature_size = x.size()
        h0 = zeros(num_layers, self.hidden_size, device=x.device)
        c0 = zeros(num_layers, self.hidden_size, device=x.device)
        x, (h, c) = self.rnn(x, (h0, c0))
        x = self.mlp(x[-1, :])
        return x



class CustomCNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lstm_dropout=0.0, bi=True):
        super(CustomCNNModel, self).__init__()
        pool = nn.MaxPool2d(kernel_size=2, stride=2)
        relu = nn.ReLU(inplace=True)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            relu,
            pool,
            nn.Conv2d(32, 64, 3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            relu,
            pool,
            nn.Conv2d(64, 128, 3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            relu,
            pool,
            nn.Conv2d(128, 256, 3, stride=1, padding='same'),
            nn.BatchNorm2d(256),
            relu,
            pool,
            nn.Conv2d(256, 512, 3, stride=1, padding='same'),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi = bi

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=lstm_dropout, batch_first=True, bidirectional=bi)

        if self.bi:
            hidden_size *= 2
        
        self.mlp = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.cnn(x)

        x = x.view(-1, 3, 512)
        
        num_layers = 2*self.num_layers if self.bi else self.num_layers
        batch_size, seq_len, feature_size = x.size()

        h0 = torch.zeros(num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(num_layers, batch_size, self.hidden_size, device=x.device)

        x, (h, c) = self.rnn(x, (h0, c0))

        x = self.mlp(x[:, -1, :])
        return x

    def encode(self, x):
        assert((3, 224, 224) == x.shape)
        #x = self.cnn(x)
        x = self.cnn(x.unsqueeze(0)).view(-1)
        assert(Size([512]) == x.shape)
        return x

    def predict_from_encoding(self, x):
        assert((3, 512) == x.shape)
        num_layers = 2*self.num_layers if self.bi else self.num_layers
        seq_len, feature_size = x.size()
        h0 = zeros(num_layers, self.hidden_size, device=x.device)
        c0 = zeros(num_layers, self.hidden_size, device=x.device)
        x, (h, c) = self.rnn(x, (h0, c0))
        x = self.mlp(x[-1, :])
        return x