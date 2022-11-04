"""Module to parse generated samples."""
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.atoms import pmg_to_atoms
from jarvis.core.lattice import Lattice
from jarvis.core.specie import atomic_numbers_to_symbols
from jarvis.db.jsonutils import dumpjson
from jarvis.analysis.structure.spacegroup import Spacegroup3D
from collections import Counter
from pymatgen.core.structure import Structure
import pandas as pd

csv_path = "/wrk/knc6/CDVAE/c2db/cdvae/data/j2d/train.csv"
df = pd.read_csv(csv_path)
df["spg_numbs"] = df["cif"].apply(
    lambda x: pmg_to_atoms(Structure.from_str(x, fmt="cif")).get_spacegroup[0]
)

print("TRAIN SPG", Counter(df["spg_numbs"].values))
print()
gen_path = "/wrk/knc6/CDVAE/c2db/cdvae/HYDRA_JOBS_J2D/singlerun/2022-11-03/j2d/eval_gen.pt"
x = torch.load(gen_path)
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
j_atomic_structures = []
spg_numbs = []
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
    spg_numb = Spacegroup3D(atoms).space_group_number
    spg_numbs.append(spg_numb)

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
    j_atomic_structures.append(atoms)

print("SPG Gen", Counter(spg_numbs))

print(atoms)
print()
print(j_atomic_structures[0])
print()
print(j_atomic_structures[1])
print()
print(j_atomic_structures[2])
print()
print(j_atomic_structures[3])
print()
print(j_atomic_structures[10])
dumpjson(data=atomic_structures, filename="eval_gen2d.json")
