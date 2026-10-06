"""Use a trained Burgers FNO to predict u(x,1) from u(x,0).

Demo: python predict_burgers.py
Your curve: python predict_burgers.py --input initial.npy --output my_prediction
The .npy file contains u(x,0) on the training grid, without duplicating x=1.
Accepted shapes: (N,), (N,1), or (B,N,1). Values must be real and unnormalized.
The demo uses a reference solver to compare answers; --input does not call it.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from fno_1d_model import FNO1d


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    equation = checkpoint['metadata']['equation']
    if (equation['viscosity'] != 0.01 or equation['final_time'] != 1.0
            or equation['initial_time'] != 0.0 or equation['domain_length'] != 1.0
            or equation['boundary'] != 'periodic'):
        raise ValueError('This tool expects a periodic Burgers checkpoint with nu=0.01 and T=1')
    model = FNO1d(**checkpoint['architecture'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, checkpoint['metadata']


def read_initial(path, n):
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray) or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError('Use a .npy file of real numbers')
    if array.ndim == 1:
        array = array[None, :, None]
    elif array.ndim == 2 and array.shape[-1] == 1:
        array = array[None, :, :]
    if array.ndim != 3 or array.shape[1:] != (n, 1) or len(array) == 0:
        raise ValueError(f'Expected ({n},), ({n},1), or (B,{n},1) on the training grid')
    initial = torch.from_numpy(array.astype(np.float32, copy=True))
    if not torch.isfinite(initial).all():
        raise ValueError('Initial values must be finite')
    return initial


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=directory / 'results' / 'burgers_fno.pt')
    parser.add_argument('--input', type=Path, help='Optional .npy initial curve; otherwise generate a demo')
    parser.add_argument('--output', type=Path, default=directory / 'predictions')
    parser.add_argument('--seed', type=int, default=9002)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error('Run train_burgers.py first, or supply --checkpoint with an existing Burgers model')
    if (args.output / 'predicted_u1.npy').exists():
        parser.error('This output already contains predictions; choose another --output folder')
    torch.set_num_threads(2)
    model, metadata = load_model(args.checkpoint)
    n = metadata['data_metadata']['sampling']['grid_points']
    if args.input:
        initial = read_initial(args.input, n)
        reference = None
    else:
        # Only this demonstration branch computes a reference answer.
        from prepare_burgers_data import make_pairs
        initial, reference = make_pairs(count=1, n=n, seed=args.seed,
                                       solver_n=metadata['data_metadata']['sampling']['solver_grid_points'])
    with torch.no_grad():
        prediction = model(initial)
    if not torch.isfinite(prediction).all():
        raise RuntimeError('Model produced non-finite predictions')
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'initial.npy', initial.numpy())
    np.save(args.output / 'predicted_u1.npy', prediction.numpy())
    with (args.output / 'prediction.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        header = ['example', 'x', 'initial_u0', 'predicted_u1']
        writer.writerow(header + (['reference_u1'] if reference is not None else []))
        for i in range(len(initial)):
            for j in range(n):
                row = [i, j/n, initial[i,j,0].item(), prediction[i,j,0].item()]
                writer.writerow(row + ([reference[i,j,0].item()] if reference is not None else []))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Matplotlib unavailable; .npy and CSV predictions were saved.')
    else:
        fig, axis = plt.subplots(figsize=(7, 4), layout='constrained')
        grid = np.arange(n) / n
        axis.plot(grid, initial[0,:,0], color='0.6', label='Initial curve, t=0')
        if reference is not None:
            axis.plot(grid, reference[0,:,0], color='#147d73', linewidth=2.5, label='Reference, t=1')
        axis.plot(grid, prediction[0,:,0], '--', color='#bd532f', linewidth=2, label='FNO prediction, t=1')
        axis.set(xlabel='x', ylabel='u', title='Burgers FNO: first input curve')
        axis.legend()
        axis.grid(alpha=0.2)
        fig.savefig(args.output / 'prediction.png', dpi=160)
        plt.close(fig)
    if reference is not None:
        error = torch.linalg.vector_norm(prediction-reference) / torch.linalg.vector_norm(reference).clamp_min(1e-8)
        print(f'Demo relative L2 error: {error.item():.2%}')
    print('Prediction shape:', tuple(prediction.shape))
    print('Saved predictions:', (args.output / 'predicted_u1.npy').resolve())


if __name__ == '__main__':
    main()
