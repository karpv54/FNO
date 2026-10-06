"""Train, evaluate, save, and reload a Burgers FNO.

Problem: u_t + u*u_x = 0.01*u_xx, periodic x in [0,1), u(x,0) -> u(x,1).
Uses fno_1d_model.FNO1d and the prepare_burgers_data data loaders.
Run: python train_burgers.py --epochs 128
"""

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from fno_1d_model import FNO1d
from prepare_burgers_data import make_loaders, make_pairs


def make_optimizer(model, learning_rate):
    """Adam must receive the parameters belonging to this model."""
    # Optimize the parameters belonging to this model.
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_one_epoch(model, loader, optimizer):
    """Visit the training examples once; return their weighted mean batch MSE."""
    # Switch the model to training mode (this alone does not train it).
    model.train()
    loss_sum, count = 0.0, 0
    for batch_u0, batch_u1 in loader:
        # Clear previous gradients with optimizer.zero_grad().
        optimizer.zero_grad()
        # Predict final profiles from the initial profiles.
        prediction = model(batch_u0)
        # Compare predictions with the target profiles using mean squared error.
        loss = F.mse_loss(prediction, batch_u1)
        if not torch.isfinite(loss):
            raise RuntimeError('Training loss is not finite; check your learning rate and data')
        # Compute gradients by calling loss.backward().
        loss.backward()
        # Update the weights by calling optimizer.step().
        optimizer.step()
        # account for a smaller final batch. item() is for logging only.
        loss_sum += loss.item() * batch_u0.size(0)
        count += batch_u0.size(0)
    if count == 0:
        raise ValueError('Training loader must contain examples')
    return loss_sum / count


def evaluate(model, loader):
    """Measure error without changing weights or recording gradients."""
    # Switch the model to evaluation mode.
    model.eval()
    mse_sum, relative_sum, worst, count = 0.0, 0.0, 0.0, 0
    # eval() and no_grad() do different jobs.
    with torch.no_grad():
        for batch_u0, batch_u1 in loader:
            prediction = model(batch_u0)
            if not torch.isfinite(prediction).all():
                raise RuntimeError('Evaluation produced non-finite predictions')
            relative = relative_l2(prediction, batch_u1)
            mse_sum += F.mse_loss(prediction, batch_u1).item() * batch_u0.size(0)
            relative_sum += relative.sum().item()
            worst = max(worst, relative.max().item())
            count += batch_u0.size(0)
    if count == 0:
        raise ValueError('Evaluation loader must contain examples')
    return {'mse': mse_sum / count, 'mean_relative_l2': relative_sum / count,
            'max_relative_l2': worst}


def snapshot_weights(model):
    """Freeze a copy; later optimizer steps must not change this snapshot."""
    # Copy the tensors so later updates cannot change the snapshot.
    return copy.deepcopy(model.state_dict())


def save_checkpoint(path, model, architecture, metadata):
    """Save an inference checkpoint containing weights and their settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {'model_state': model.state_dict(),
                  'architecture': architecture, 'metadata': metadata}
    # Save the weights alongside the architecture and metadata.
    torch.save(checkpoint, path)


def load_checkpoint(path):
    """Rebuild the same architecture and insert the saved trained weights."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = FNO1d(**checkpoint['architecture'])
    # Restore the trained weights.
    model.load_state_dict(checkpoint['model_state'])
    # Switch the reloaded model to evaluation mode.
    model.eval()
    return model, checkpoint['metadata']


ARCHITECTURE = {'width': 16, 'modes': 12, 'depth': 3}
EXPECTED_EQUATION = {'name': 'viscous Burgers', 'viscosity': 0.01,
                     'initial_time': 0.0, 'final_time': 1.0,
                     'domain_length': 1.0, 'boundary': 'periodic'}


def relative_l2(prediction, target):
    """One error per curve; 0.02 means 2% relative L2 error, not 98% accuracy."""
    if prediction.shape != target.shape or target.ndim != 3:
        raise ValueError('Expected matching (examples,N,1) tensors')
    error_norm = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    target_norm = torch.linalg.vector_norm(target.flatten(1), dim=1)
    return error_norm / target_norm.clamp_min(1e-8)


def read_dataset(path):
    if not path.is_file():
        raise FileNotFoundError(f'Run prepare_burgers_data.py first: missing {path}')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    metadata = saved['metadata']
    if metadata['equation'] != EXPECTED_EQUATION or metadata['normalization'] != 'none':
        raise ValueError('Use the unnormalized periodic Burgers dataset from prepare_burgers_data.py')
    partitions = saved['splits']
    if set(partitions) != {'train', 'validation', 'test'}:
        raise ValueError('Expected train, validation, and test partitions')
    for name, (u0, u1) in partitions.items():
        if (u0.shape != u1.shape or u0.ndim != 3 or u0.size(-1) != 1
                or len(u0) != metadata['split_sizes'][name] or len(u0) == 0
                or u0.size(1) != metadata['sampling']['grid_points']):
            raise ValueError(f'Invalid {name} tensor shapes')
        if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (u0, u1)):
            raise ValueError(f'Expected finite float32 {name} tensors')
    return partitions, metadata


def save_plots(output, history, u0, target, prediction):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Matplotlib is unavailable; numerical results are still saved.', flush=True)
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    axes[0].loglog([r['epoch'] for r in history], [r['train_mse'] for r in history], label='Training')
    axes[0].loglog([r['epoch'] for r in history], [r['validation_mse'] for r in history], label='Validation')
    axes[0].set(xlabel='Epoch (log scale)', ylabel='Mean squared error (log scale)',
                title='Learning from the Burgers data')
    axes[0].legend()
    n = u0.size(1)
    grid = torch.arange(n) / n
    axes[1].plot(grid, u0[0, :, 0], color='0.65', label='Initial curve, t=0')
    axes[1].plot(grid, target[0, :, 0], color='#147d73', linewidth=2.5, label='Reference, t=1')
    axes[1].plot(grid, prediction[0, :, 0], '--', color='#bd532f', linewidth=1.8, label='FNO, t=1')
    axes[1].set(xlabel='x', ylabel='u', title='A new initial condition')
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(output / 'training_and_prediction.png', dpi=160)
    plt.close(fig)


def run_training(dataset_path, output, epochs, learning_rate):
    if (output / 'burgers_fno.pt').exists():
        raise FileExistsError('A model already exists here. Use --output results_try2 for another run.')
    partitions, data_metadata = read_dataset(dataset_path)
    # Use a fixed seed for reproducible model initialization.
    torch.manual_seed(42)
    loaders = make_loaders(partitions, batch_size=32, seed=82)
    model = FNO1d(**ARCHITECTURE)
    optimizer = make_optimizer(model, learning_rate)
    # gradually reduce the learning rate, after each full epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate/10)
    initial_validation = evaluate(model, loaders['validation'])
    print(f'Untrained validation relative L2: {initial_validation["mean_relative_l2"]:.2%}', flush=True)
    best_error, best_epoch, best_state = float('inf'), None, None
    history = []
    for epoch in range(1, epochs + 1):
        train_mse = train_one_epoch(model, loaders['train'], optimizer)
        validation = evaluate(model, loaders['validation'])
        history.append({'epoch': epoch, 'train_mse': train_mse,
                        'validation_mse': validation['mse'],
                        'validation_relative_l2': validation['mean_relative_l2']})
        if validation['mean_relative_l2'] < best_error:
            best_error = validation['mean_relative_l2']
            best_state = snapshot_weights(model)
            best_epoch = epoch
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f'Epoch {epoch:3d}/{epochs}: train MSE={train_mse:.6f}; '
                  f'validation relative L2={validation["mean_relative_l2"]:.2%}', flush=True)

    # Select only with validation. The test set is first evaluated after this choice.
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders['test'])
    test_u0, test_u1 = partitions['test']
    identity_error = relative_l2(test_u0, test_u1).mean().item()
    mean_prediction = test_u0.mean(dim=1, keepdim=True).expand_as(test_u1)
    mean_error = relative_l2(mean_prediction, test_u1).mean().item()
    metadata = {'equation': EXPECTED_EQUATION, 'data_metadata': data_metadata,
                'dataset_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
                'architecture': ARCHITECTURE, 'epochs': epochs, 'best_epoch': best_epoch,
                'initial_learning_rate': learning_rate, 'batch_size': 32,
                'model_seed': 42, 'loader_seed': 82, 'device': 'cpu',
                'torch_version': str(torch.__version__), 'checkpoint_purpose': 'inference'}
    save_checkpoint(output / 'burgers_fno.pt', model, ARCHITECTURE, metadata)
    restored, _ = load_checkpoint(output / 'burgers_fno.pt')
    # A fresh curve for demonstration; it does not participate in model selection.
    new_u0, new_u1 = make_pairs(count=1, n=test_u0.size(1), seed=9001,
                              solver_n=data_metadata['sampling']['solver_grid_points'])
    with torch.no_grad():
        prediction = restored(new_u0)
    new_error = relative_l2(prediction, new_u1).item()
    report = {'initial_validation': initial_validation, 'best_validation_relative_l2': best_error,
              'best_epoch': best_epoch, 'test': test_metrics,
              'copy_input_baseline_relative_l2': identity_error,
              'spatial_mean_baseline_relative_l2': mean_error,
              'new_example_seed': 9001, 'new_example_relative_l2': new_error,
              'metadata': metadata}
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    torch.save({'u0': new_u0, 'reference_u1': new_u1, 'predicted_u1': prediction},
               output / 'new_example.pt')
    with (output / 'new_prediction.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['x', 'initial_u0', 'reference_u1', 'predicted_u1'])
        n = new_u0.size(1)
        for j in range(n):
            writer.writerow([j/n, new_u0[0,j,0].item(), new_u1[0,j,0].item(), prediction[0,j,0].item()])
    save_plots(output, history, new_u0, new_u1, prediction)
    print(f'Best validation epoch: {best_epoch}', flush=True)
    print(f'Test mean relative L2: {test_metrics["mean_relative_l2"]:.2%}', flush=True)
    print(f'Test worst relative L2: {test_metrics["max_relative_l2"]:.2%}', flush=True)
    print(f'Copy-input baseline: {identity_error:.2%}; mean baseline: {mean_error:.2%}', flush=True)
    print(f'New example relative L2: {new_error:.2%}', flush=True)
    print(f'Model saved and reloaded: {(output / "burgers_fno.pt").resolve()}', flush=True)


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--dataset', type=Path, default=directory / 'data' / 'burgers_pairs.pt')
    parser.add_argument('--output', type=Path, default=directory / 'results')
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('epochs and learning-rate must be positive and finite')
    torch.set_num_threads(2)
    run_training(args.dataset, args.output, args.epochs, args.learning_rate)


if __name__ == '__main__':
    main()
