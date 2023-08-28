import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as T
from datetime import datetime
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Training on device {device}.")


def get_data_flattened():
  data_path = '../_00_data/j_cifar10/'

  # class_names = [
  #   'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
  # ]

  # input.shape: torch.Size([-1, 3, 32, 32]) --> torch.Size([-1, 3072])
  transformed_cifar10 = datasets.CIFAR10(
    data_path, train=True, download=True, transform=transforms.Compose([
      transforms.ToTensor(), transforms.Normalize(
        mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616)
      ),
      T.Lambda(lambda x: torch.flatten(x))
    ])
  )

  transformed_cifar10_val = datasets.CIFAR10(
    data_path, train=False, download=True, transform=transforms.Compose([
      transforms.ToTensor(), transforms.Normalize(
        mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616)
      ),
      T.Lambda(lambda x: torch.flatten(x))
    ])
  )

  train_data_loader = DataLoader(
    dataset=transformed_cifar10, batch_size=wandb.config.batch_size, shuffle=True
  )
  validation_data_loader = DataLoader(
    dataset=transformed_cifar10_val, batch_size=wandb.config.batch_size
  )

  return train_data_loader, validation_data_loader


def get_model_and_optimizer():
  class MyModel(nn.Module):
    def __init__(self, n_input, n_output):
      super().__init__()

      self.model = nn.Sequential(
        nn.Linear(n_input, wandb.config.n_hidden_unit_list[0]),
        nn.Sigmoid(),
        nn.Linear(wandb.config.n_hidden_unit_list[0], wandb.config.n_hidden_unit_list[1]),
        nn.Sigmoid(),
        nn.Linear(wandb.config.n_hidden_unit_list[1], n_output),
      )

    def forward(self, x):
      x = self.model(x)
      return x

  # 3 * 32 * 32 = 3072
  my_model = MyModel(n_input=3072, n_output=10).to(device)
  optimizer = optim.SGD(my_model.parameters(), lr=wandb.config.learning_rate)

  return my_model, optimizer


def training_loop(model, optimizer, train_data_loader, validation_data_loader):
  n_epochs = wandb.config.epochs
  loss_fn = nn.CrossEntropyLoss()  # Use a built-in loss function

  for epoch in range(1, n_epochs + 1):
    loss_train = 0.0
    num_corrects_train = 0
    num_train_samples = 0
    for idx, train_batch in enumerate(train_data_loader):
      input, target = train_batch
      input = input.to(device)
      target = target.to(device)

      output = model(input)
      loss = loss_fn(output, target)
      loss_train += loss.item()
      predicted = torch.argmax(output, dim=1)
      num_corrects_train += int((predicted == target).sum())
      num_train_samples += len(train_batch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    loss_validation = 0.0
    num_corrects_validation = 0
    num_validation_samples = 0
    with torch.no_grad():
      for idx, validation_batch in enumerate(validation_data_loader):
        input, target = validation_batch
        input = input.to(device)
        target = target.to(device)

        output = model(input)
        loss_validation += loss_fn(output, target).item()
        predicted = torch.argmax(output, dim=1)
        num_corrects_validation += int((predicted == target).sum())
        num_validation_samples += len(validation_batch)

    if epoch == 1 or epoch % 10 == 0:
      print(
        f"[Epoch {epoch}] "
        f"Training loss {loss_train / num_train_samples:.4f}, "
        f"Training accuracy {num_corrects_train / num_train_samples:.4f} | "
        f"Validation loss {loss_validation / num_validation_samples:.4f}, "
        f"Validation accuracy {num_corrects_validation / num_validation_samples:.4f}"
      )

    wandb.log({
      "Epoch": epoch,
      "Training loss": loss_train / num_train_samples,
      "Training accuracy": num_corrects_train / num_train_samples,
      "Validation loss": loss_validation / num_validation_samples,
      "Validation accuracy": num_corrects_validation / num_validation_samples,
    })


def main():
  current_time_str = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

  config = {
    'epochs': 10_000,
    'learning_rate': 1e-4,
    'batch_size': 256,
    'n_hidden_unit_list': [20, 20],
  }

  wandb.init(
    mode="disabled",
    project="dnn_cifar10",
    notes="cifar10 experiment",
    tags=["dnn", "cifar10"],
    name=current_time_str,
    config=config
  )

  train_data_loader, validation_data_loader = get_data_flattened()

  linear_model, optimizer = get_model_and_optimizer()

  wandb.watch(linear_model)

  training_loop(
    model=linear_model,
    optimizer=optimizer,
    train_data_loader=train_data_loader,
    validation_data_loader=validation_data_loader
  )
  wandb.finish()


if __name__ == "__main__":
  main()