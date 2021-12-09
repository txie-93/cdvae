import os
import pytest
import numpy as np
import torch
from torch_scatter.scatter import scatter
from cdvae.common import data_utils


def test_lattice_params_matrix():
    a, b, c = 4., 3., 2.
    alpha, beta, gamma = 120., 90., 90.

    matrix = data_utils.lattice_params_to_matrix(a, b, c, alpha, beta, gamma)
    result = data_utils.lattice_matrix_to_params(matrix)

    assert np.allclose([a, b, c, alpha, beta, gamma], result)


def test_lattice_params_matrix2():
    matrix = [[3.96686600e+00, 0.00000000e+00, 2.42900487e-16],
              [-2.42900487e-16, 3.96686600e+00, 2.42900487e-16],
              [0.00000000e+00, 0.00000000e+00, 5.73442000e+00]]
    matrix = np.array(matrix)
    params = data_utils.lattice_matrix_to_params(matrix)
    result = data_utils.lattice_params_to_matrix(*params)

    assert np.allclose(matrix, result)


def test_lattice_params_to_matrix_torch():
    lengths = np.array([[4., 3., 2.], [1, 3, 2]])
    angles = np.array([[120., 90., 90.], [57., 130., 85.]])

    lengths_and_angles = np.concatenate([lengths, angles], axis=-1)

    matrix0 = data_utils.lattice_params_to_matrix(
        *lengths_and_angles[0].tolist())
    matrix1 = data_utils.lattice_params_to_matrix(
        *lengths_and_angles[1].tolist())

    true_matrix = np.stack([matrix0, matrix1], axis=0)

    torch_matrix = data_utils.lattice_params_to_matrix_torch(
        torch.Tensor(lengths), torch.Tensor(angles))

    assert np.allclose(true_matrix, torch_matrix.numpy(), atol=1e-5)


def test_frac_cart_conversion():
    num_atoms = torch.LongTensor([4, 3, 2, 5])
    lengths = torch.rand(num_atoms.size(0), 3) * 4
    angles = torch.rand(num_atoms.size(0), 3) * 60 + 60
    frac_coords = torch.rand(num_atoms.sum(), 3)

    cart_coords = data_utils.frac_to_cart_coords(
        frac_coords, lengths, angles, num_atoms)

    inverted_frac_coords = data_utils.cart_to_frac_coords(
        cart_coords, lengths, angles, num_atoms)

    assert torch.allclose(frac_coords, inverted_frac_coords)


def test_get_pbc_distances():
    frac_coords = torch.Tensor([[0.2, 0.2, 0.], [0.6, 0.8, 0.8],
                                [0.2, 0.2, 0.], [0.6, 0.8, 0.8]])
    edge_index = torch.LongTensor([[1, 0], [0, 0], [2, 3]]).T
    lengths = torch.Tensor([[1., 1., 2.], [1., 2., 1.]])
    angles = torch.Tensor([[90., 90., 90.], [90., 90., 90.]])
    to_jimages = torch.LongTensor([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
    num_nodes = torch.LongTensor([2, 2])
    num_edges = torch.LongTensor([2, 1])

    out = data_utils.get_pbc_distances(
        frac_coords, edge_index, lengths, angles, to_jimages, num_nodes, num_edges)

    true_distances = torch.Tensor([1.7549928774784245, 1., 1.2])

    assert torch.allclose(true_distances, out['distances'])


def test_get_pbc_distances_cart():
    frac_coords = torch.Tensor([[0.2, 0.2, 0.], [0.6, 0.8, 0.8],
                                [0.2, 0.2, 0.], [0.6, 0.8, 0.8]])
    edge_index = torch.LongTensor([[1, 0], [0, 0], [2, 3]]).T
    lengths = torch.Tensor([[1., 1., 2.], [1., 2., 1.]])
    angles = torch.Tensor([[90., 90., 90.], [90., 90., 90.]])
    to_jimages = torch.LongTensor([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
    num_nodes = torch.LongTensor([2, 2])
    num_edges = torch.LongTensor([2, 1])

    cart_coords = data_utils.frac_to_cart_coords(
        frac_coords, lengths, angles, num_nodes)

    out = data_utils.get_pbc_distances(
        cart_coords, edge_index, lengths, angles, to_jimages, num_nodes, num_edges,
        coord_is_cart=True)

    true_distances = torch.Tensor([1.7549928774784245, 1., 1.2])

    assert torch.allclose(true_distances, out['distances'])


@pytest.mark.parametrize('max_radius,max_neighbors', [
    (3, 20),  # test small cutoff radius
    (6, 12),  # test if max_neighbors is satisfied
])
def test_radius_graph_pbc(max_radius, max_neighbors):
    from cdvae.pl_data.dataset import CrystDataset
    from torch_geometric.data import Batch
    from cdvae.common.data_utils import get_scaler_from_data_list

    dir_path = os.path.dirname(os.path.realpath(__file__))
    test_file_path = os.path.join(dir_path, 'test_data.csv')
    dataset = CrystDataset(
        name='test',
        path=test_file_path,
        prop='formation_energy_per_atom',
        niggli=True,
        primitive=False,
        graph_method='crystalnn',
        lattice_scale_method='scale_length',
        preprocess_workers=2,
    )

    scaler = get_scaler_from_data_list(
        dataset.cached_data,
        key=dataset.prop)

    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)

    edge_index, unit_cell, num_neighbors_image = data_utils.radius_graph_pbc_wrapper(
        batch, radius=max_radius, max_num_neighbors_threshold=max_neighbors,
        device=batch.num_atoms.device)

    batch.edge_index = edge_index
    batch.to_jimages = unit_cell
    batch.num_bonds = num_neighbors_image

    outputs = data_utils.get_pbc_distances(
        batch.frac_coords,
        batch.edge_index,
        batch.lengths,
        batch.angles,
        batch.to_jimages,
        batch.num_atoms,
        batch.num_bonds,
    )

    assert outputs['distances'].shape[0] > 0
    assert outputs['distances'].max() <= max_radius
    row, col = outputs['edge_index']

    for i in range(batch.num_nodes):
        # can only assert col (index_i) satisfy the max_neighbor requirement
        assert torch.nonzero(col == i, as_tuple=True)[
            0].shape[0] <= max_neighbors


def test_compute_volume():
    batched_lattice = torch.Tensor([
        [[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]],
        [[1., 1., 0.], [1., -1., 0.], [0., 0., 1.]],
    ])
    true_volumes = torch.Tensor([6., 2.])

    results = data_utils.compute_volume(batched_lattice)

    assert torch.allclose(true_volumes, results)
