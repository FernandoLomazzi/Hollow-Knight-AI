import torchvision.models as models
from torch import cuda, zeros, Size, nn

class ResnetModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, lstm_dropout=0.0, bi=True):
        super(ResnetModel, self).__init__()
        self.resnet18 = models.resnet18(weights='DEFAULT').to("cuda" if cuda.is_available() else "cpu")
        self.resnet18.fc = nn.Identity()
        #for param in self.resnet18.parameters():
        #    param.requires_grad = False
        
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
        # 64, 7, 256
        """
        doing x.view(batch_size*4, 3, H, W) and then when each image has been processed 
        by the CNN you will have shape (batch_size*4, C, H_new, W_new) 
        and you can reshape this to the lstm by doing x.view(batch_size, 4, -1) if you’re using batch_first=True for the lstm. The lstm will then process them in a sequence.
        """
        x = x.view(-1, 3, 224, 224)
        x = self.resnet18(x)

        x = x.view(-1, 3, 512)
        
        num_layers = 2*self.num_layers if self.bi else self.num_layers
        batch_size, seq_len, feature_size = x.size()

        h0 = zeros(num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = zeros(num_layers, batch_size, self.hidden_size, device=x.device)

        x, (h, c) = self.rnn(x, (h0, c0))

        #x, _ = self.rnn(x, h0)
        # x -> [batch_size, sequence_length, hidden_size]
        # h -> [num_layers, batch_size, hidden_size]
        #print(x[:, -1, :] == h[-1, :, :])
        x = self.mlp(x[:, -1, :])
        return x

    def encode(self, x):
        assert((3, 224, 224) == x.shape)
        x = self.resnet18(x.unsqueeze(0)).squeeze(dim=0)
        assert(Size([512]) == x.shape)
        return x

    def predict_from_encoding(self, x):
        assert((3, 512) == x.shape)
        #x = x.view(-1, 3, 512)
        num_layers = 2*self.num_layers if self.bi else self.num_layers
        seq_len, feature_size = x.size()
        h0 = zeros(num_layers, self.hidden_size, device=x.device)
        c0 = zeros(num_layers, self.hidden_size, device=x.device)
        x, (h, c) = self.rnn(x, (h0, c0))
        x = self.mlp(x[-1, :])
        return x