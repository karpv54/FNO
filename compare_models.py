"""Compare the FNO, an equally parameterized affine baseline, and forward Euler.

Run after prepare_burgers_data.py:
    python compare_models.py --epochs 128
Repeat plots and timings from saved weights:
    python compare_models.py --skip-training
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from burgers_data_generator import initial_profile, sample_coefficients
from euler_burgers import euler_burgers, stable_euler_steps
from fno_1d_model import FNO1d
from linear_model import LinearFNO1d, parameter_count
from prepare_burgers_data import make_loaders
from train_burgers import ARCHITECTURE, evaluate, make_optimizer, read_dataset, relative_l2, train_one_epoch


COLORS = {'FNO': '#087f8c', 'Linear': '#b95730', 'Euler': '#6850a1'}
LOSS_METRICS = {'train_mse': ('Training MSE', 'Mean squared error'),
                'validation_mse': ('Validation MSE', 'Mean squared error'),
                'validation_relative_l2': ('Validation relative L2', 'Mean relative L2 error')}


def fit_power_law(rows, name, metric):
    """Fit measured positive values in log space; do not change the data."""
    values = np.array([row[name][metric] for row in rows], dtype=float)
    epochs = np.array([row['epoch'] for row in rows], dtype=float)
    valid = np.isfinite(values) & (values > 0) & (epochs > 0)
    if len(rows) < 3 or not valid.all() or len(np.unique(epochs)) != len(rows):
        return {'available': False, 'reason': 'need at least 3 distinct epochs with positive finite values'}
    x, y = np.log(epochs), np.log(values)
    slope, intercept = np.polyfit(x, y, 1)
    total = np.sum((y - y.mean())**2)
    r2 = 1 - np.sum((y - (slope * x + intercept))**2) / max(total, 1e-30)
    return {'available': True, 'gamma': -float(slope),
            'coefficient': float(np.exp(intercept)), 'log_log_r2': float(r2),
            'start_epoch': int(epochs[0]), 'end_epoch': int(epochs[-1]),
            'points': len(rows)}


def two_window_fits(history, name):
    """Split epochs >=10 equally by count, matching the original slope check."""
    rows = [row for row in history if row['epoch'] >= 10]
    midpoint = (len(rows) + 1) // 2
    windows = {'early': rows[:midpoint], 'late': rows[midpoint:]}
    return {metric: {label: fit_power_law(window, name, metric)
                     for label, window in windows.items()} for metric in LOSS_METRICS}


def train_models(partitions, epochs, learning_rate, output, dataset_sha256):
    torch.manual_seed(42)
    models = {'FNO': FNO1d(**ARCHITECTURE), 'Linear': LinearFNO1d(**ARCHITECTURE)}
    # Start with identical weights, mini-batch orders, and optimizer settings.
    models['Linear'].load_state_dict(copy.deepcopy(models['FNO'].state_dict()))
    loaders = {name: make_loaders(partitions, batch_size=32, seed=82) for name in models}
    optimizers = {name: make_optimizer(model, learning_rate) for name, model in models.items()}
    schedulers = {name: torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate / 10)
        for name, optimizer in optimizers.items()}
    best = {name: {'error': float('inf'), 'epoch': None, 'state': None} for name in models}
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        row = {'epoch': epoch}
        for name, model in models.items():
            loss = train_one_epoch(model, loaders[name]['train'], optimizers[name])
            validation = evaluate(model, loaders[name]['validation'])
            row[name] = {'train_mse': loss, 'validation_mse': validation['mse'],
                         'validation_relative_l2': validation['mean_relative_l2']}
            if validation['mean_relative_l2'] < best[name]['error']:
                best[name] = {'error': validation['mean_relative_l2'], 'epoch': epoch,
                              'state': copy.deepcopy(model.state_dict())}
            schedulers[name].step()
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs}: " + '; '.join(
                f"{name} train MSE={row[name]['train_mse']:.6g}, "
                f"validation L2={row[name]['validation_relative_l2']:.2%}"
                for name in models), flush=True)
    training = {'epochs': epochs, 'learning_rate': learning_rate, 'batch_size': 32,
                'dataset_sha256': dataset_sha256,
                'model_seed': 42, 'loader_seed': 82,
                'wall_seconds_both_models': time.perf_counter() - start,
                'selection': 'minimum validation mean relative L2',
                'best_epoch': {name: best[name]['epoch'] for name in models}}
    for name, model in models.items():
        model.load_state_dict(best[name]['state'])
        model.eval()
    torch.save({'architecture': ARCHITECTURE, 'training': training,
                'states': {name: model.state_dict() for name, model in models.items()}},
               output / 'comparison_models.pt')
    (output / 'loss_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return models, history, training


def load_models(output):
    checkpoint = torch.load(output / 'comparison_models.pt', map_location='cpu', weights_only=True)
    models = {'FNO': FNO1d(**checkpoint['architecture']),
              'Linear': LinearFNO1d(**checkpoint['architecture'])}
    for name, model in models.items():
        model.load_state_dict(checkpoint['states'][name])
        model.eval()
    history = json.loads((output / 'loss_history.json').read_text(encoding='utf-8'))
    return models, history, checkpoint['training']


def power_law_evidence(history, name):
    """A descriptive check, not a theorem about optimization convergence."""
    rows = [row for row in history if row['epoch'] >= 10]
    if len(rows) < 30:
        return {'supported': False, 'reason': 'fewer than 30 epochs after epoch 9'}
    x = np.log([row['epoch'] for row in rows])
    y = np.log([row[name]['train_mse'] for row in rows])
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    total = np.sum((y - y.mean())**2)
    r2 = 1 - np.sum(residual**2) / max(total, 1e-30)
    halves = np.array_split(np.arange(len(rows)), 2)
    gammas = [-np.polyfit(x[ids], y[ids], 1)[0] for ids in halves]
    gamma = -float(slope)
    consistent = gamma > 0.05 and abs(gammas[0] - gammas[1]) <= 0.25 * gamma
    return {'supported': bool(r2 >= 0.98 and consistent), 'gamma': gamma,
            'log_log_r2': float(r2), 'half_window_gammas': [float(g) for g in gammas],
            'start_epoch': rows[0]['epoch'], 'end_epoch': rows[-1]['epoch'],
            'coefficient': float(np.exp(intercept)),
            'criteria': 'R2 >= 0.98, gamma > 0.05, half-window slopes differ by <= 25% of gamma'}


def save_figure(fig, output, name):
    fig.savefig(output / f'{name}.png', dpi=180)
    fig.savefig(output / f'{name}.pdf')
    plt.close(fig)


def plot_losses(history, window_fits, output):
    epochs = [row['epoch'] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), layout='constrained')
    for axis, (metric, (title, ylabel)) in zip(axes, LOSS_METRICS.items()):
        for name in ('FNO', 'Linear'):
            axis.loglog(epochs, [row[name][metric] for row in history],
                        color=COLORS[name], linewidth=2, label=f'{name}: measured')
        # Show both FNO trends independently of the linear baseline's plateau.
        # These are descriptive fits, not evidence of two exact scaling laws.
        for label, style in (('early', '--'), ('late', ':')):
            fit = window_fits['FNO'][metric][label]
            if not fit['available']:
                continue
            x = np.geomspace(fit['start_epoch'], fit['end_epoch'], 100)
            axis.loglog(x, fit['coefficient'] * x**(-fit['gamma']), style,
                        color='#182e42', linewidth=2.2,
                        label=f"FNO {fit['start_epoch']}-{fit['end_epoch']}: "
                              f"gamma={fit['gamma']:.2f}, R2={fit['log_log_r2']:.3f}")
        early, late = (window_fits['FNO'][metric][key] for key in ('early', 'late'))
        if early['available'] and late['available']:
            boundary = np.sqrt(early['end_epoch'] * late['start_epoch'])
            axis.axvline(boundary, color='0.5', linewidth=1, alpha=0.4)
        axis.set(xlabel='Epoch E (log scale)', ylabel=ylabel + ' (log scale)', title=title)
        axis.legend(fontsize=8, loc='upper right')
        axis.grid(alpha=0.2, which='both')
    fig.suptitle('Measured loss and FNO fits: C E^(-gamma) in two training windows\n'
                 'Separate fitted trends; logarithmic axes do not establish a power law', fontsize=12)
    save_figure(fig, output, 'loss_comparison')


@torch.no_grad()
def compare_accuracy(models, partitions, output):
    u0, target = partitions['test']
    inputs = u0[..., 0].numpy().astype(np.float64)
    reference = target[..., 0].numpy().astype(np.float64)
    predictions = {name: model(u0) for name, model in models.items()}
    model_metrics = {}
    for name, prediction in predictions.items():
        errors = relative_l2(prediction, target)
        model_metrics[name] = {'mean_relative_l2': errors.mean().item(),
                               'max_relative_l2': errors.max().item(),
                               'mse': (prediction - target).square().mean().item()}
    # All 64 test curves use the same steps, grid, and initial values.
    base_steps = stable_euler_steps(inputs)
    euler_rows = []
    euler_predictions = []
    for factor in (1, 2, 4):
        steps = factor * base_steps
        prediction = euler_burgers(inputs, steps)
        errors = np.linalg.norm(prediction - reference, axis=-1) / np.maximum(
            np.linalg.norm(reference, axis=-1), 1e-8)
        euler_rows.append({'steps': steps, 'dt': 1 / steps,
                           'mean_relative_l2': float(errors.mean()),
                           'max_relative_l2': float(errors.max()),
                           'mse': float(np.mean((prediction - reference)**2)),
                           'mean_absolute_mass_error': float(np.abs(
                               prediction.mean(-1) - inputs.mean(-1)).mean())})
        euler_predictions.append(prediction)
        print(f"Euler: {steps} steps, dt={1/steps:.6g}, test L2={errors.mean():.2%}", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout='constrained')
    x = [row['dt'] for row in euler_rows]
    y = [100 * row['mean_relative_l2'] for row in euler_rows]
    axes[0].plot(x, y, 'o-', color=COLORS['Euler'], linewidth=2, label='Euler')
    axes[0].axhline(100 * model_metrics['FNO']['mean_relative_l2'], color=COLORS['FNO'],
                    linestyle='--', linewidth=2, label='FNO')
    for row, error in zip(euler_rows, y):
        axes[0].annotate(f"{row['steps']} steps\n{error:.2f}%", (row['dt'], error),
                         xytext=(0, 12), textcoords='offset points', ha='center', fontsize=9)
    axes[0].set_xscale('log')
    axes[0].invert_xaxis()
    axes[0].set(xlabel='Euler time step dt (smaller to the right)',
                ylabel='Mean relative L2 error (%)',
                title=f'Accuracy at t=1, fixed grid N={u0.size(1)}')
    axes[0].margins(y=0.35)
    grid = np.arange(u0.size(1)) / u0.size(1)
    axes[1].plot(grid, reference[0], color='#30343b', linewidth=2.5, label='Cole-Hopf target')
    axes[1].plot(grid, predictions['FNO'][0, :, 0], '--', color=COLORS['FNO'],
                 linewidth=2, label='FNO')
    axes[1].plot(grid, euler_predictions[-1][0], ':', color=COLORS['Euler'],
                 linewidth=2.5, label=f'Euler, {euler_rows[-1]["steps"]} steps')
    axes[1].set(xlabel='x', ylabel='u(x,1)', title='First held-out profile')
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    save_figure(fig, output, 'euler_accuracy')
    np.savez(output / 'accuracy_predictions.npz', initial=inputs, reference=reference,
             fno=predictions['FNO'][..., 0].numpy(), linear=predictions['Linear'][..., 0].numpy(),
             euler=np.stack(euler_predictions), dt=np.array(x))
    return {'model_test_metrics': model_metrics, 'euler_test_metrics': euler_rows,
            'test_examples': len(u0), 'grid_points': u0.size(1),
            'note': 'Smaller dt removes time error, but Rusanov spatial error remains on this fixed grid.'}


def time_call(function, repeats):
    function()  # Warm up kernels and allocations; do not time this call.
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {'median_ms': float(np.median(samples)), 'min_ms': float(np.min(samples)),
            'q25_ms': float(np.quantile(samples, 0.25)),
            'q75_ms': float(np.quantile(samples, 0.75)), 'samples_ms': samples}


@torch.inference_mode()
def benchmark(models, grids, repeats, output):
    # Sample the same continuous initial condition at every resolution.
    coefficients, mean = sample_coefficients(1, seed=701)
    rows = []
    for n in grids:
        initial = initial_profile(coefficients, mean, n)
        tensor = torch.from_numpy(initial.astype(np.float32)).unsqueeze(-1)
        steps = stable_euler_steps(initial)
        row = {'grid_points': n, 'euler_steps': steps, 'euler_dt': 1 / steps}
        for name, model in models.items():
            row[name] = time_call(lambda: model(tensor), max(30, repeats))
        row['Euler'] = time_call(lambda: euler_burgers(initial, steps), repeats)
        rows.append(row)
        print(f"Timing N={n}: " + ', '.join(f"{name}={row[name]['median_ms']:.3f} ms"
              for name in ('FNO', 'Linear', 'Euler')), flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout='constrained')
    ns = np.array(grids, dtype=float)
    for name in ('FNO', 'Linear', 'Euler'):
        values = np.array([row[name]['median_ms'] for row in rows])
        lower = np.array([row[name]['q25_ms'] for row in rows])
        upper = np.array([row[name]['q75_ms'] for row in rows])
        axes[0].plot(ns, values, 'o-', color=COLORS[name], linewidth=2, label=name)
        axes[0].fill_between(ns, lower, upper, color=COLORS[name], alpha=0.15)
    axes[0].set_xscale('log', base=2)
    axes[0].set_yscale('log')
    axes[0].set(xlabel='Spatial grid points N', ylabel='Milliseconds per profile',
                title='Measured CPU prediction time (median, IQR)')
    axes[0].set_xticks(grids, [str(n) for n in grids])
    growth = ns / ns[0]
    axes[1].loglog(ns, growth, ':', color='#80868b', label='O(N): one Euler step')
    axes[1].loglog(ns, growth * np.log2(ns) / np.log2(ns[0]),
                   color=COLORS['FNO'], linewidth=2, label='O(N log N): FNO / linear')
    axes[1].loglog(ns, growth**3, color=COLORS['Euler'], linewidth=2,
                   label='O(N^3): full explicit Euler solve')
    axes[1].set(xlabel='Spatial grid points N', ylabel=f'Growth relative to N={grids[0]}',
                title='Algorithmic scaling guides (not timing fits)')
    for axis in axes:
        axis.legend(fontsize=9)
        axis.grid(alpha=0.2, which='both')
    fig.suptitle('Batch 1; t=0 to t=1; training and target generation excluded', fontsize=12)
    save_figure(fig, output, 'prediction_time')
    with (output / 'timings.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['grid_points', 'euler_steps', 'FNO_ms', 'Linear_ms', 'Euler_ms'])
        for row in rows:
            writer.writerow([row['grid_points'], row['euler_steps']] +
                             [row[name]['median_ms'] for name in ('FNO', 'Linear', 'Euler')])
    return {'rows': rows, 'batch_size': 1, 'torch_threads': torch.get_num_threads(),
            'torch_version': str(torch.__version__), 'numpy_version': np.__version__,
            'machine': platform.machine(), 'platform': platform.platform(),
            'model_dtype': 'float32 / complex64', 'euler_dtype': 'float64',
            'warmup_calls': 1, 'model_repeats': max(30, repeats), 'euler_repeats': repeats,
            'scope': 'preloaded initial arrays and weights; no training, I/O, reference solver, or plotting',
            'complexity': {'FNO': 'O(N log N) for fixed width, depth, and modes',
                           'Linear': 'O(N log N); same Fourier architecture',
                           'Euler_step': 'O(N)',
                           'Euler_to_T1': 'O(N^3) as diffusion limits dt to O(N^-2)'},
            'note': 'Timings depend on the machine, dtype, Python overhead, and library kernels; they do not prove an asymptotic class.'}


def write_summary(report, output):
    metrics = report['accuracy']['model_test_metrics']
    lines = ['# Burgers model comparison', '',
             'One seeded experiment; same data splits, initial weights, Adam settings, and mini-batch order.', '',
             'The linear baseline retains the Fourier architecture and removes every GELU. '
             'It is affine in u0; it is not a dense regression layer.', '',
             f"Both models have **{report['real_scalar_parameters']:,} real scalar weights** "
             '(complex weights count twice).', '',
             '| Method | Test mean relative L2 | Test worst relative L2 |',
             '| --- | ---: | ---: |']
    for name in ('FNO', 'Linear'):
        lines.append(f"| {name} | {metrics[name]['mean_relative_l2']:.2%} | {metrics[name]['max_relative_l2']:.2%} |")
    for row in report['accuracy']['euler_test_metrics']:
        lines.append(f"| Euler, {row['steps']} steps (dt={row['dt']:.6g}) | "
                     f"{row['mean_relative_l2']:.2%} | {row['max_relative_l2']:.2%} |")
    lines += ['', '## Loss curves', '',
              'Training MSE, validation MSE, and validation mean relative L2 use logarithmic axes '
              'for both epoch and error. Recorded values and the MSE training objective are unchanged.', '',
              'Two descriptive FNO fits use epochs >=10, split into equal-count early and late windows. '
              'Each fit is C * epoch^(-gamma); gamma and log-space R2 describe that window only. '
              'They do not establish two exact laws. The linear baseline remains visible as measured data.', '',
              '| FNO metric | Epochs | gamma | Log-log R2 |',
              '| --- | --- | ---: | ---: |']
    for metric, windows in report['two_window_power_law_fits']['FNO'].items():
        for fit in windows.values():
            if fit['available']:
                lines.append(f"| {LOSS_METRICS[metric][0]} | {fit['start_epoch']}-{fit['end_epoch']} | "
                             f"{fit['gamma']:.3f} | {fit['log_log_r2']:.3f} |")
    lines += ['', 'The original single-exponent consistency check is retained in comparison.json. '
              'A failed check can coexist with useful descriptive fits over shorter windows.', '',
              '## Euler accuracy', '',
              f"The three time steps solve the same {report['accuracy']['test_examples']} held-out profiles "
              f"on the fixed {report['accuracy']['grid_points']}-point grid. "
              'The solver uses forward Euler in time, a conservative Rusanov advection stencil, '
              'and centered physical diffusion. Decreasing dt alone does not remove its first-order spatial error.', '',
              '## Prediction timing', '',
              'Times measure one initial profile to t=1 on this machine, excluding training, disk I/O, '
              'plotting, and Cole-Hopf target generation. FNO and linear use float32/complex64; '
              'Euler uses float64. Shaded bands are interquartile ranges over repeated calls.', '',
              'For fixed FNO width, depth, and retained modes, the FFT gives O(N log N) scaling. '
              'Euler costs O(N) per step. For fixed viscosity and final time, the explicit diffusion '
              'restriction dt=O(N^-2) makes the full Euler solve O(N^3). '
              'These are algorithmic scaling estimates, not classes inferred from the measured timing curves.', '',
              'Sources: [FNO complexity](https://neuraloperator.github.io/dev/theory_guide/fno.html#complexity); '
              '[finite differences and diffusion stability](https://ocw.mit.edu/courses/18-086-mathematical-methods-for-engineers-ii-spring-2006/5db29e69494eb09a26f7224d43adc6f6_am54.pdf).', '',
              '## Reproduce', '', '```bash', 'python prepare_burgers_data.py',
              'python compare_models.py --epochs 128', '```', '',
              'To redraw figures and repeat timings using the saved comparison weights:', '',
              '```bash', 'python compare_models.py --skip-training', '```', '']
    (output / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=directory / 'data' / 'burgers_pairs.pt')
    parser.add_argument('--output', type=Path, default=directory / 'comparison_results')
    parser.add_argument('--epochs', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--timing-repeats', type=int, default=5)
    parser.add_argument('--grids', type=int, nargs='+', default=[64, 128, 256, 512, 1024])
    args = parser.parse_args()
    if (args.epochs < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or args.timing_repeats < 3 or len(args.grids) < 3
            or args.grids != sorted(set(args.grids)) or min(args.grids) < 32):
        parser.error('Use positive epochs/lr, >=3 repeats, and >=3 increasing unique grid sizes >=32')
    torch.set_num_threads(2)
    partitions, metadata = read_dataset(args.dataset)
    dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.skip_training:
        models, history, training = load_models(args.output)
        if training['dataset_sha256'] != dataset_sha256:
            raise ValueError('Saved comparison used another dataset; choose a new output and train again')
    else:
        if (args.output / 'comparison_models.pt').exists():
            parser.error('A comparison already exists: use --skip-training or a different --output')
        models, history, training = train_models(partitions, args.epochs, args.learning_rate,
                                                  args.output, dataset_sha256)
    counts = {name: parameter_count(model) for name, model in models.items()}
    if counts['FNO'] != counts['Linear']:
        raise RuntimeError('The baseline and FNO parameter counts do not match')
    evidence = {name: power_law_evidence(history, name) for name in models}
    window_fits = {name: two_window_fits(history, name) for name in models}
    plot_losses(history, window_fits, args.output)
    accuracy = compare_accuracy(models, partitions, args.output)
    timing = benchmark(models, args.grids, args.timing_repeats, args.output)
    report = {'real_scalar_parameters': counts['FNO'], 'architecture': ARCHITECTURE,
              'training': training, 'dataset_metadata': metadata, 'power_law_check': evidence,
              'two_window_power_law_fits': window_fits,
              'show_power_law_fits': any(fit['available']
                  for windows in window_fits['FNO'].values() for fit in windows.values()),
              'fit_interpretation': 'Descriptive FNO window fits, not two established scaling laws',
              'accuracy': accuracy, 'timing': timing}
    (args.output / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    write_summary(report, args.output)
    print('Comparison saved:', args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
