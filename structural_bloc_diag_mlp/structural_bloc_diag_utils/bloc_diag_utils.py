import torch
import torch.nn as nn
import torch.optim as optim



def translate_nonzero_blocks(input_str):
  string_nonzero_blocks = input_str.split(' | ')
  string_nonzero_blocks = [expr.split(', ') for expr in string_nonzero_blocks]
  string_nonzero_blocks = [item for sublist in string_nonzero_blocks for item in sublist] # Now, string_nonzero_blocks is a list of separated strings
  row_blocks = [eval(expr) for expr in string_nonzero_blocks[::2]] # even indexed elements
  col_blocks = [eval(expr) for expr in string_nonzero_blocks[1::2]] # odd indexed elements

  return row_blocks, col_blocks


def train_model_per_epoch(model, loader, optimizer, criterion, scaling_off_diag_loss, calculate_accuracy=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in loader:
        optimizer.zero_grad()
        output = model(data.view(data.size(0), -1))
        off_diag_loss = model.get_off_diag_loss()
        loss = criterion(output, target) + scaling_off_diag_loss * off_diag_loss
        loss.backward()
        optimizer.step()

        if calculate_accuracy:
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    if calculate_accuracy:
        train_loss = running_loss / len(loader)
        train_accuracy = correct / total
        return train_loss, train_accuracy


def train_and_test_evaluate(model, train_loader, test_loader, epochs, learning_rate, scaling_off_diag_loss, scheduler_step_size=20, scheduler_gamma=0.5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    train_accuracy = 0
    test_accuracy_before_dropping = 0
    test_accuracy_off_diag_dropped = 0
    for epoch in range(epochs):
        # Determine if this epoch needs accuracy calculation
        calculate_accuracy = (epoch + 1) == epochs # calculate_accuracy is a Boolean

        if calculate_accuracy:
            train_loss, train_accuracy = train_model_per_epoch(model, train_loader, optimizer, criterion, scaling_off_diag_loss, calculate_accuracy)
            print(f"Epoch {epoch + 1}/{epochs} - Train loss: {train_loss:.4f}, Train accuracy: {100*train_accuracy:.2f}%")


            ####### evaluate accuracy on the test set #######
            test_accuracy_before_dropping = evaluate_model_accuracy(model, test_loader)
            print(f"Test accuracy, before dropping off diagonal entries: {100*test_accuracy_before_dropping:.2f}%")

            model_off_diag_dropped = model.copy_model_drop_off_diag()
            test_accuracy_off_diag_dropped = evaluate_model_accuracy(model_off_diag_dropped, test_loader)
            print(f"Test accuracy, after dropping off diagonal entries: {100*test_accuracy_off_diag_dropped:.2f}%")
        else:
            # Perform training without calculating accuracy
            train_model_per_epoch(model, train_loader, optimizer, criterion, scaling_off_diag_loss, calculate_accuracy)

        scheduler.step()

    return train_accuracy, test_accuracy_before_dropping, test_accuracy_off_diag_dropped


def train_and_val_evaluate(model, train_loader, val_loader, epochs, learning_rate, scaling_off_diag_loss, scheduler_step_size=20, scheduler_gamma=0.5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    n_epochs = []
    train_accuracies = []
    val_accuracies_before_dropping = []
    val_accuracies_off_diag_dropped = []
    for epoch in range(epochs):
        # Determine if this epoch needs accuracy calculation
        calculate_accuracy = (epoch + 1) % 5 == 0 # calculate_accuracy is a Boolean

        if calculate_accuracy:
            train_loss, train_accuracy = train_model_per_epoch(model, train_loader, optimizer, criterion, scaling_off_diag_loss, calculate_accuracy)
            print(f"Epoch {epoch + 1}/{epochs} - Train loss: {train_loss:.4f}, Train accuracy: {100 * train_accuracy:.2f}%")
            
            n_epochs.append(epoch)
            train_accuracies.append(train_accuracy)


            ####### evaluate accuracy on the validation set #######
            val_accuracy_model_before_dropping = evaluate_model_accuracy(model, val_loader)
            print(f"Validation accuracy, before dropping off diagonal entries: {100 * val_accuracy_model_before_dropping:.2f}%")
            val_accuracies_before_dropping.append(val_accuracy_model_before_dropping)

            model_off_diag_dropped = model.copy_model_drop_off_diag()
            val_accuracy_model_off_diag_dropped = evaluate_model_accuracy(model_off_diag_dropped, val_loader)
            print(f"Validation accuracy, after dropping off diagonal entries: {100 * val_accuracy_model_off_diag_dropped:.2f}%")
            val_accuracies_off_diag_dropped.append(val_accuracy_model_off_diag_dropped)
        else:
            # Perform training without calculating accuracy
            train_model_per_epoch(model, train_loader, optimizer, criterion, scaling_off_diag_loss, calculate_accuracy)

        scheduler.step()

    return n_epochs, train_accuracies, val_accuracies_before_dropping, val_accuracies_off_diag_dropped


def evaluate_model_accuracy(model, loader):
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data.view(data.size(0), -1))
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy