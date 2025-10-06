import torch



def test_sbd_model(model, testloader, criterion, device):
  
    model.eval()
    correct, total, loss, counter = 0, 0, 0, 0
    
    with torch.no_grad():
        for (images, labels) in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()
            counter += 1
    
    return loss / counter, correct / total