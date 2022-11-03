"""Module to parse generated samples."""
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.lattice import Lattice
from jarvis.core.specie import atomic_numbers_to_symbols
from jarvis.db.jsonutils import dumpjson

# from jarvis.core.atoms import Atoms, pmg_to_atoms
# from pymatgen.core.structure import Structure
# from pymatgen.core.lattice import Lattice as Lat

# python cdvae/run.py data=carbon expname=carbon
#  model.predict_property=True

# python scripts/evaluate.py --model_path
# /wrk/knc6/CDVAE/cdvae/HYDRA_JOBS/singlerun/2022-10-22/carbon
# --tasks gen

# x = torch.load("HYDRA_JOBS/singlerun/2022-10-22/carbon/eval_gen.pt")
x = torch.load("HYDRA_JOBS/singlerun/2022-11-02/custom/eval_gen.pt")
num_atoms = x["num_atoms"]
atom_types = x["atom_types"]
frac_coords = x["frac_coords"]
lengths = x["lengths"]
angles = x["angles"]
index_list = torch.cumsum(num_atoms[0], dim=0).numpy().tolist()
indice_tuples = []
for i, ii in enumerate(index_list):
    if i == 0:
        tup = [0, index_list[i] - 1]
    else:
        tup = [index_list[i - 1] - 1, index_list[i] - 1]
    indice_tuples.append(tup)

atomic_structures = []
for id_needed in range(num_atoms.shape[1]):
    id_fracs = frac_coords[0].numpy()[
        indice_tuples[id_needed][0] : indice_tuples[id_needed][1]
    ]
    id_atom_types = atom_types[0].numpy()[
        indice_tuples[id_needed][0] : indice_tuples[id_needed][1]
    ]
    id_lengths = lengths[0].numpy()[id_needed]
    id_angles = angles[0].numpy()[id_needed]
    lat = Lattice.from_parameters(
        id_lengths[0],
        id_lengths[1],
        id_lengths[2],
        id_angles[0],
        id_angles[1],
        id_angles[2],
    ).matrix
    atoms = Atoms(
        lattice_mat=lat,
        elements=atomic_numbers_to_symbols(id_atom_types),
        coords=id_fracs,
        cartesian=False,
    )
    # print()
    # print()
    # print()
    # print("jarvis\n", atoms)
    # struct = Structure(
    #    lattice=Lat.from_parameters(
    #        id_lengths[0],
    #        id_lengths[1],
    #        id_lengths[2],
    #        id_angles[0],
    #        id_angles[1],
    #        id_angles[2],
    #    ),
    #    species=id_atom_types,
    #    coords=id_fracs,
    #    coords_are_cartesian=False,
    # )
    # atoms = pmg_to_atoms(struct)
    # print("pmg\n", atoms)
    # print()
    # print()
    # print()

    atomic_structures.append(atoms.to_dict())

print(atoms)
dumpjson(data=atomic_structures, filename="eval_gen2d.json")
