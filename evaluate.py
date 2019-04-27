from model import Model
import argparse
import json
import torch

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def load_data(path, batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )

    val_data = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=transform)
    test_data = datasets.ImageFolder('{}/{}/test'.format(path, 'supervised'), transform=transform)
    data_loader_val = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    data_loader_test = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return data_loader_val, data_loader_test

def evaluate(model, data_loader, device, split, top_k=5):
    model.eval()
    n_samples = 0.
    n_correct_top_1 = 0
    n_correct_top_k = 0

    for img, target in data_loader:
        img, target = img.to(device), target.to(device)
        batch_size = img.size(0)
        n_samples += batch_size

        # Forward
        output = model(img)

        # Top 1 accuracy
        top_1_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        n_correct_top_1 += top_1_pred.eq(target.view_as(top_1_pred)).sum().item()

        # Top k accuracy
        top_k_ind = torch.from_numpy(output.data.cpu().numpy().argsort()[:, -top_k:]).long()
        target_top_k = target.view(-1, 1).expand(batch_size, top_k)
        n_correct_top_k += (top_k_ind == target_top_k.cpu()).sum().item()

    # Accuracy
    top_1_acc = n_correct_top_1/n_samples
    top_k_acc = n_correct_top_k/n_samples

    # Log
    print('{} top 1 accuracy: {:.4f}'.format(split, top_1_acc))
    print('{} top {} accuracy: {:.4f}'.format(split, top_k, top_k_acc))


if __name__ == '__main__':
    # Define arguments
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='location of data')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--model_path', type=str, default='weights.pth',
                        help='location of model weights')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1008, metavar='S',
                        help='random seed')
    # Parse arguments
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(json.dumps(args.__dict__, sort_keys=True, indent=4) + '\n')
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # Load model
    # model = Model().load_model(args.model_path)
    model = Model()

    # Load data
    data_loader_val, data_loader_test = load_data(args.data_dir, args.batch_size)


    # Evaluate
    evaluate(model, data_loader_val, args.device, 'Validation')
    evaluate(model, data_loader_test, args.device, 'Test')
