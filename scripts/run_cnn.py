# Load CTCF sequences using the function students already implemented
ctcf_sequences = load_encode_sequences("ENCFF029THO.bed.gz", "chr21.fa", "chr21")

# Prepare positive examples (CTCF binding sites)
S = []
for seq in ctcf_sequences:
    if len(seq) <= 500:
        S.append((seq.upper(), 1))

# Using the pyfaidx library to extract sequences from the file "chr21.fa".
# Randomly sample 3000 sequences with lengths between 350 and 500
# from the fasta sequence. These sequences are uppercased and
# tagged with a label '0'.
seq = Fasta("chr21.fa")[0]
for _ in range(3000):
    i = np.random.randint(len(seq))
    l = np.random.randint(350, 500)
    t = str(seq[i:i+l])
    S.append((t.upper(), 0))  # Random sequences labeled as 0

# Split into train and test set
np.random.shuffle(S)
split = int(0.8 * len(S))  # 80/20 split
train = S[:split]
test = S[split:]
print(f"Train: {len(train)} \tTest: {len(test)}")

# One-hot encoding function
def one_hot(seq):
    while len(seq) < 500:
        seq += 'N'
    nd = {
        'A': [1.0,0.0,0.0,0.0],
        'C': [0.0,1.0,0.0,0.0],
        'G': [0.0,0.0,1.0,0.0],
        'T': [0.0,0.0,0.0,1.0],
        'N': [0.0,0.0,0.0,0.0]
    }
    return np.array([nd[x] for x in seq]).T

class SeqDataset(Dataset):
    def __init__(self, data):
        """
        Create a PyTorch dataset from sequence data.

        Args:
            data: List of (sequence, label) tuples
        """

        seq = [torch.tensor(one_hot(x), dtype=torch.float32) for x, _ in data]
        labels = [int(y) for _, y in data]

        self.seq = torch.stack(seq, dim=0)
        self.labels = torch.tensor(labels, dtype=torch.float32)
      
    def __getitem__(self, idx):
        return self.seq[idx], self.labels[idx]

    def __len__(self):
        return self.seq.shape[0]

# Create data loaders
train_loader = DataLoader(SeqDataset(train), batch_size=256)
test_loader = DataLoader(SeqDataset(test), batch_size=256)

def train_model(model, train_loader, optimizer):
    """
    Train the model for one epoch.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_items = 0

    criterion = nn.BCELoss()

    for input, label in train_loader:

        optimizer.zero_grad()
        output = model(input)
        output = output.squeeze(1)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = (output > 0.5).long()
        total_correct += (preds==label.long()).sum().item()
        total_items += label.size(0)
    return total_loss/len(train_loader), total_correct/total_items

def test_model(model, test_loader):
    """
    Evaluate model on test data.

    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()

    total_loss = 0
    total_correct = 0
    total_items = 0
    criterion = nn.BCELoss()

    for input, label in test_loader:
      torch.no_grad()
      output = model(input)
      output = output.squeeze(1)
      loss = criterion(output, label)
      total_loss += loss.item()
      preds = (output > 0.5).long()
      total_correct += (preds == label.long()).sum().item()
      total_items += input.size(0)

    return total_loss/len(train_loader), total_correct/total_items

# Define CNN architecture
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

# Train and evaluate
train_loss, train_acc = train_model(model, train_loader, optimizer)
test_loss, test_acc = test_model(model, test_loader)

print(f"Train loss: {train_loss:.4f}\tTrain accuracy: {train_acc:.4f}")
print(f"Test loss: {test_loss:.4f}\tTest accuracy: {test_acc:.4f}")
