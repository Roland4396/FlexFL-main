#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    return iid(dataset, num_users)


def mnist_noniid(dataset, num_users, alpha=100):
    """MNIST Non-IID using Dirichlet distribution"""
    return build_non_iid_by_dirichlet(dataset, num_users, alpha)


def fashion_mnist_iid(dataset, num_users):
    return iid(dataset, num_users)


def fashion_mnist_noniid(dataset, num_users, alpha=100):
    """Fashion-MNIST Non-IID using Dirichlet distribution"""
    return build_non_iid_by_dirichlet(dataset, num_users, alpha)


def cifar_iid(dataset, num_users):
    return iid(dataset, num_users)


def cifar_noniid(dataset, num_users, alpha=100):
    """
    CIFAR Non-IID using Dirichlet distribution

    Args:
        dataset: CIFAR dataset
        num_users: Number of clients
        alpha: Dirichlet concentration parameter (integer, range 1-100+)
               - alpha=1: highly non-IID
               - alpha=10: moderate non-IID
               - alpha=100: near IID (default)
    """
    return build_non_iid_by_dirichlet(dataset, num_users, alpha)


def cifar100_iid(dataset, num_users):
    return iid(dataset, num_users)


def cifar100_noniid(dataset, num_users, alpha=100):
    """CIFAR-100 Non-IID using Dirichlet distribution"""
    return build_non_iid_by_dirichlet(dataset, num_users, alpha)


def svhn_iid(dataset, num_users):
    return iid(dataset, num_users)


def svhn_noniid(dataset, num_users, alpha=100):
    """SVHN Non-IID using Dirichlet distribution"""
    return build_non_iid_by_dirichlet(dataset, num_users, alpha)


def iid(dataset, num_users):
    """IID data partitioning"""
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    for i in range(num_users):
        dict_users[i] = np.array(list(dict_users[i])).tolist()
    return dict_users


def build_non_iid_by_dirichlet(dataset, n_workers, non_iid_alpha):
    """
    Advanced Non-IID partitioning using Dirichlet distribution
    Source: SmartFL (adapted from FedDF paper)

    This method creates realistic non-IID splits by:
    1. Using Dirichlet distribution to control label distribution per client
    2. Automatic balancing to ensure minimum data per client
    3. Multi-stage partitioning for scalability

    Args:
        dataset: Dataset with .targets attribute (list of labels)
        n_workers: Number of clients/workers
        non_iid_alpha: Concentration parameter (integer, range 1-100+)
                       - alpha=1: highly non-IID
                       - alpha=10: moderate non-IID
                       - alpha=100: near IID (default in SmartFL)

    Returns:
        dict: {client_id: [list of sample indices]}
    """
    import math

    # Get dataset info
    targets = np.array(dataset.targets)
    num_indices = len(targets)
    num_classes = len(np.unique(targets))

    # Create indices2targets array
    indices2targets = np.array([[idx, target] for idx, target in enumerate(targets)])

    # Initialize random state for reproducibility
    random_state = np.random.RandomState(1)

    # Parameters for multi-stage partitioning
    n_auxi_workers = min(10, n_workers)  # Auxiliary workers for scalability

    # Random shuffle
    random_state.shuffle(indices2targets)

    # Partition indices into splits for scalability
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]

    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index: (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    # Build idx_batch using Dirichlet distribution
    idx_batch = []
    remaining_workers = n_workers

    for _targets in splitted_targets:
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # Use auxi_workers for this subset
        _n_workers = min(n_auxi_workers, remaining_workers)
        remaining_workers = remaining_workers - n_auxi_workers

        # Ensure minimum size constraint
        min_size = 0
        min_required = int(0.50 * _targets_size / _n_workers)

        while min_size < min_required:
            _idx_batch = [[] for _ in range(_n_workers)]

            # For each class, use Dirichlet to distribute samples
            for _class in range(num_classes):
                # Get indices for this class
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                if len(idx_class) == 0:
                    continue

                try:
                    # Sample from Dirichlet distribution
                    proportions = random_state.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )

                    # Balance: prevent workers from getting too much data
                    proportions = np.array([
                        p * (len(idx_j) < _targets_size / _n_workers)
                        for p, idx_j in zip(proportions, _idx_batch)
                    ])
                    proportions = proportions / proportions.sum()

                    # Split indices according to proportions
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[:-1]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(_idx_batch, np.split(idx_class, proportions))
                    ]

                    # Check minimum size
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min(sizes) if sizes else 0

                except (ZeroDivisionError, ValueError):
                    # If split fails, retry
                    min_size = 0
                    continue

        idx_batch += _idx_batch

    # Convert to dictionary format
    dict_users = {i: np.array(v, dtype='int64').tolist() for i, v in enumerate(idx_batch)}

    return dict_users


if __name__ == '__main__':

    trans = transforms.Compose([transforms.ToTensor()])
    dataset_train = datasets.SVHN('../data/svhn/', split='train', download=True, transform=trans)
    num = 100
    d = svhn_noniid(dataset_train, num, alpha=10)
    for user_idx in d:
        print(user_idx)
        print([dataset_train[img_idx][1] for img_idx in d[user_idx]])
