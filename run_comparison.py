"""Install missing packages and run the comparison from this script's folder."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys


DIRECTORY = Path(__file__).resolve().parent


def run(script, *arguments):
    subprocess.run([sys.executable, str(DIRECTORY / script), *map(str, arguments)],
                   cwd=DIRECTORY, check=True)


def install_if_needed():
    # Check imports and versions in the same interpreter that runs the project.
    check = r"""
import re
from importlib.metadata import version
import torch, numpy, matplotlib
for name, minimum, maximum in [('torch',(2,8),3), ('numpy',(1,26),3), ('matplotlib',(3,8),4)]:
    match = re.match(r'(\d+)\.(\d+)', version(name))
    value = tuple(map(int, match.groups()))
    if value < minimum or value[0] >= maximum:
        raise RuntimeError(name + ' needs a compatible version')
"""
    ready = subprocess.run([sys.executable, '-c', check], cwd=DIRECTORY,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ready.returncode == 0:
        print('Required packages are already installed.', flush=True)
        return
    print('Installing compatible packages. This may take a few minutes.', flush=True)
    has_pip = subprocess.run([sys.executable, '-m', 'pip', '--version'],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if has_pip.returncode:
        subprocess.run([sys.executable, '-m', 'ensurepip', '--upgrade'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                    str(DIRECTORY / 'requirements.txt')], cwd=DIRECTORY, check=True)


def choose_output(dataset):
    """Reuse matching trained weights; keep earlier runs if the dataset changed."""
    output = DIRECTORY / 'comparison_results'
    checkpoint_path = output / 'comparison_models.pt'
    if not checkpoint_path.exists():
        return output, False
    if (output / 'loss_history.json').exists():
        import torch
        try:
            saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if saved['training']['dataset_sha256'] == hashlib.sha256(dataset.read_bytes()).hexdigest():
                return output, True
        except (OSError, KeyError, ValueError, RuntimeError):
            pass
    index = 2
    while (DIRECTORY / f'comparison_results_{index}').exists():
        index += 1
    return DIRECTORY / f'comparison_results_{index}', False


def main():
    if sys.version_info < (3, 10):
        raise RuntimeError('This project needs Python 3.10 or newer.')
    required = ['compare_models.py', 'linear_model.py', 'euler_burgers.py',
                'fno_1d_model.py', 'spectral_layer.py', 'prepare_burgers_data.py',
                'burgers_data_generator.py', 'train_burgers.py', 'requirements.txt']
    missing = [name for name in required if not (DIRECTORY / name).is_file()]
    if missing:
        raise RuntimeError('Extract all the files into one folder. Missing: ' + ', '.join(missing))
    print('Burgers comparison: FNO, linear model, and Euler', flush=True)
    print(f'Python {sys.version.split()[0]}', flush=True)
    install_if_needed()
    dataset = DIRECTORY / 'data' / 'burgers_pairs.pt'
    if not dataset.exists():
        print('Preparing the dataset...', flush=True)
        run('prepare_burgers_data.py')
    output, reuse = choose_output(dataset)
    print('Reusing trained comparison weights.' if reuse else 'Training both models for 128 epochs...', flush=True)
    arguments = ['--dataset', dataset, '--output', output]
    arguments += ['--skip-training'] if reuse else ['--epochs', '128']
    run('compare_models.py', *arguments)
    print('\nDone. Your three graphs are in:', output, flush=True)
    if os.name == 'nt':
        os.startfile(str(output))


if __name__ == '__main__':
    try:
        main()
    except (subprocess.CalledProcessError, OSError, RuntimeError) as error:
        print('\nThe comparison stopped:', error, flush=True)
        sys.exit(1)
