"""Prepare paired Burgers data for u(x,0) -> u(x,1).

Equation: u_t + u*u_x = 0.01*u_xx, x in [0,1), periodic boundaries.
The solver in burgers_data_generator.py generates the target profiles.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from burgers_data_generator import sample_coefficients, initial_profile, solve_burgers


VISCOSITY = 0.01
FINAL_TIME = 1.0


def make_pairs(count, n, seed, solver_n=512):
    """Return u0 and u1, both real float32 tensors of shape (count,n,1)."""
    if count < 1 or n <= 16 or solver_n < n or solver_n % n != 0:
        raise ValueError('Require count > 0, n > 16, and solver_n a multiple of n')

    # Coefficients and means specify the same continuous input functions
    # regardless of which uniform grid we sample them on.
    coefficients, mean = sample_coefficients(count, seed)
    stride = solver_n // n

    # Evaluate the initial profile on the model grid.
    # It contains the initial values on the model grid, shape (count,n).
    u0_numpy = initial_profile(coefficients, mean, n)

    # Solve on the finer grid at the specified viscosity and final time.
    u1_fine = solve_burgers(coefficients, mean, n=solver_n, viscosity=VISCOSITY, final_time=FINAL_TIME)

    # Sample every stride-th spatial value, retaining all examples: (count,n).
    u1_numpy = u1_fine[:, ::stride]

    # Convert to float32 tensors and add the final size-one channel axis.
    u0 = torch.from_numpy(u0_numpy).float().unsqueeze(-1)

    # Apply the same conversion to the target profiles.
    u1 = torch.from_numpy(u1_numpy).float().unsqueeze(-1)

    return u0, u1


def split_pairs(u0, u1, n_train, n_validation, seed):
    """Shuffle whole-example IDs once, then form three non-overlapping splits."""
    if u0.shape != u1.shape or u0.ndim != 3 or u0.size(-1) != 1:
        raise ValueError('Expected matching (examples,N,1) input and target tensors')
    total = u0.size(0)
    if n_train < 1 or n_validation < 1 or n_train + n_validation >= total:
        raise ValueError('Training, validation, and test sets must all be nonempty')
    generator = torch.Generator().manual_seed(seed)

    # Shuffle example IDs using the supplied seed.
    order = torch.randperm(total, generator=generator)

    # Reserve different whole examples for the three purposes.
    train_ids = order[:n_train]
    validation_ids = order[n_train:n_train + n_validation]
    test_ids = order[n_train + n_validation:]

    # Use the same IDs for each input and target; select examples along axis 0.
    partitions = {'train': (u0[train_ids], u1[train_ids]), 'validation': (u0[validation_ids], u1[validation_ids]), 'test': (u0[test_ids], u1[test_ids])}

    return partitions


def make_loader(u0, u1, batch_size, shuffle, seed):
    """Package paired examples and deliver them in mini-batches."""
    generator = torch.Generator().manual_seed(seed)

    # Keep inputs and targets paired in a TensorDataset.
    dataset = TensorDataset(u0, u1)

    # Keep the final partial batch and use the seeded generator for shuffling.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0, drop_last=False)

    return loader


def make_loaders(partitions, batch_size, seed):
    """Training shuffles paired examples; validation/test preserve their order."""
    loaders = {}
    for name, pair in partitions.items():
        # Shuffle only the training split.
        loaders[name] = make_loader(pair[0], pair[1], batch_size, shuffle=(name == 'train'), seed=seed)
    return loaders


def prepare_data():
    """Save the paired dataset, with the equation and sampling metadata."""
    settings = {'examples': 640, 'grid_points': 128, 'solver_grid_points': 512,
                'data_seed': 2026, 'split_seed': 81, 'loader_seed': 82}
    u0, u1 = make_pairs(settings['examples'], settings['grid_points'], settings['data_seed'], settings['solver_grid_points'])
    partitions = split_pairs(u0, u1, n_train=512, n_validation=64, seed=settings['split_seed'])

    equation = {'name': 'viscous Burgers', 'viscosity': 0.01, 'initial_time': 0.0,
                'final_time': 1.0, 'domain_length': 1.0, 'boundary': 'periodic'}
    metadata = {'equation': equation, 'sampling': settings,
                'split_sizes': {name: len(pair[0]) for name, pair in partitions.items()},
                'normalization': 'none', 'dtype': 'float32',
                'input_distribution': {'harmonics': 8, 'raw_coefficient_std': '1/k^2',
                    'coefficient_l1_norm': 'uniform[0.2,1.0]', 'mean': 'uniform[-0.3,0.3]'}}
    output = Path(__file__).parent / 'data'
    output.mkdir(exist_ok=True)
    dataset_path = output / 'burgers_pairs.pt'
    torch.save({'metadata': metadata, 'splits': partitions}, dataset_path)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Dataset splits:', metadata['split_sizes'])
    print('Saved dataset:', dataset_path.resolve())


if __name__ == '__main__':
    prepare_data()
