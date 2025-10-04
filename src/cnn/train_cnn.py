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
