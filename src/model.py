import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv3x3(in_channels, out_channels, stride=1):
    """easier way to define the conv layer many times"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ResidualBlock(nn.Module):
    """the residual block for residual cnn"""
    def __init__(
        self, in_channels, out_channels, stride=1, downsample=None, dropout=0.1
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return self.dropout(out)


class BidirectionalLSTM(nn.Module):
    """a more sofisticated biderictional lstm"""
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Time-distributed output

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(x)


class ASRCNNFeatureExtractor(nn.Module):
    """the cnn block that's gonna transform the spetrogram to feature maps"""
    def __init__(self, input_channels=1, dropout=0.1):
        super(ASRCNNFeatureExtractor, self).__init__()

        self.stage1 = self._make_stage(
            input_channels, 32, num_blocks=2, first_stride=(2, 1), dropout=dropout
        )
        self.stage2 = self._make_stage(
            32, 64, num_blocks=2, first_stride=(2, 2), dropout=dropout
        )
        self.stage3 = self._make_stage(
            64, 128, num_blocks=2, first_stride=(2, 1), dropout=dropout
        )
        self.output_seq_len = 104
        self.output_features = 128 * 10

    def _make_stage(
        self, in_channels, out_channels, num_blocks, first_stride=(1, 1), dropout=0.1
    ):

        layers = []

        downsample = None

        if first_stride != (1, 1) or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=first_stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        layers.append(
            ResidualBlock(
                in_channels,
                out_channels,
                stride=first_stride,
                downsample=downsample,
                dropout=dropout,
            )
        )

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dropout=dropout))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(batch_size, self.output_seq_len, self.output_features)

        return x


class ASRModel(nn.Module):
    """the whole model combined of the feature maps cnn extractor and biderictionnal lstm model 
       and then give log probability of caracters for each time step"""
    def __init__(self, num_classes, hidden_size=256, num_lstm_layers=2, dropout=0.1):
        super(ASRModel, self).__init__()

        self.cnn = ASRCNNFeatureExtractor(dropout=dropout)

        self.lstm = BidirectionalLSTM(
            input_size=self.cnn.output_features,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.cnn(x)

        x = self.lstm(x)

        x = torch.log_softmax(x, dim=-1)

        return x
