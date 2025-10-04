model = nn.Sequential(
    nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool1d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(32*249, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

optimizer = torch.optim.Adam(model.parameters())
