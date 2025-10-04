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
