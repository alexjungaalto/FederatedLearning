import argparse
import warnings
from collections import OrderedDict

import flwr as fl
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    choices=[0, 1, 2, 3, 4],
    required=True,
    type=int,
    help="Partition of the dataset divided into 5 iid partitions created artificially.",
)
parser.add_argument(
    "--server-address",
    default="127.0.0.1:8080",
    type=str,
    help="Address (IP:PORT) of Flower server."
)

##################################### Model defintion and train/eval loops #####################

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz').
    
    This is the model we'll be training in FL. Note that clients
    are not stateful by default. In Flower 1.8+ you'll be able
    to save/retrieve the state of a client."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def set_parameters(net, ndarrays):
    """This is an auxhiliary function that receives a list of
    numpy arrays and loads them as model parameters. We'll be using
    this function to update the parameters of the model in the clients
    with those sent by the server"""
    params_dict = zip(net.state_dict().keys(), ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs, device, disable_tqdm : bool):
    """Train the model on the training set.
    
    This function is no different from the typical training loop
    for supervised image classification in the centralized setting."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    for _ in range(epochs):
        for batch in tqdm(trainloader, "Training", disable=disable_tqdm):
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()


def test(net, testloader, device, disable_tqdm: bool):
    """Validate the model on the test set.

    This function is no different from the typical evaluation loop
    for supervised image classification in the centralized setting."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader, "Testing", disable=disable_tqdm):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

####################################### Data preparation ###################################

def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    # Standard set of transforms for MNIST
    transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch

def load_data(partition_id, num_partitions: int=5):
    """Download MNIST if not present and partition it into `num_partitions`. Then,
    return the partition_id-th partition wrapped into its respective train/test
    dataloaders.
    
    Please note here we decided to split each partition into train/test but this
    is optional and application specific. Also, the dataloader could be create inside
    the client if that's more appropiate in your setting."""
    fds = FederatedDataset(dataset="mnist", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2)

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, testloader, disable_tqdm):
        """Constructor of our FlowerClient"""
        self.net = Net()
        # This client will be (in Part-B) used for Simulation, so it's preferred to
        # set the device like this (as opposed to do so with a global variable defined
        # at the top of the file.)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader
        self.disable_tqdm = disable_tqdm

    def get_parameters(self, config):
        """Extract parameters from the model and return them as a list of NumPy arrays.
        This is the format that the Server and aggregation strategy knows how to operate with."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters, config):
        """Perform local training.
        The client has recevied `parameters` from the server and, optionally,
         a `config` that can be used to parameterize how fit() behaves."""
        
        # Apply parameters to this client's model
        set_parameters(self.net, parameters)
        # Train for one epoch using this client's training set
        train(self.net, self.trainloader, epochs=1, device=self.device, disable_tqdm=self.disable_tqdm)
        # Extract model parameters and return
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Perform local evaluation.
        The client has received `parameters` from the server and, optionally, 
        a `config` to parameterize this stage."""
        # Apply parameters to this client's model
        set_parameters(self.net, parameters)
        # Evaluate the model on the local evaluation/test set (note the model is not changed)
        loss, accuracy = test(self.net, self.testloader, device=self.device, disable_tqdm=self.disable_tqdm)
        # Report back to the server the results of the evaluation
        # Different applications make use of different metrics, here we use accuracy
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def get_client_fn(total_partitions: int = 5, disable_tqdm: bool = False):
    def client_fn(cid: str):
        """A function that returns a client associated with a
        data partition.
        
        This function is executed each time a client is to be spawned."""

        # Prepare training and testing data loaders for a client
        trainloader, testloader = load_data(partition_id=int(cid),
                                            num_partitions=total_partitions)
        
        # Return the client object
        return FlowerClient(trainloader, testloader, disable_tqdm).to_client()
    return client_fn



def main():

    # Parse arguments
    args = parser.parse_args()

    # Get function that spawns a client
    client_fn = get_client_fn()

    # Start Flower client
    fl.client.start_client(
        server_address=args.server_address,
        client=client_fn(args.partition_id),
    )

if __name__ == "__main__":
    main()
