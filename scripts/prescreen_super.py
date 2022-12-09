from jarvis.db.jsonutils import loadjson
from jarvis.core.atoms import Atoms
from pretrained import (
    get_multiple_predictions,
    get_prediction,
    get_figshare_model,
)
from jarvis.core.graphs import Graph
import torch
from alignn.models.alignn import ALIGNN, ALIGNNConfig

d = loadjson("eval_gensup.json")
cutoff = 8
max_neighbors = 12
filename = "/wrk/knc6/SuperConDeb/HT_DATA1053/Tc/out/checkpoint_200.pt"
output_features = 1
model = ALIGNN(ALIGNNConfig(name="alignn", output_features=output_features))
device = "cpu"
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
model.load_state_dict(torch.load(filename, map_location=device)["model"])
model.to(device)
model.eval()

form_model = get_figshare_model(
    model_name="jv_formation_energy_peratom_alignn"
)
form_model.to(device)


bg_model = get_figshare_model(model_name="jv_optb88vdw_bandgap_alignn")
bg_model.to(device)
print("size", len(d))
for i in d:
    try:
        atoms = Atoms.from_dict(i)

        g, lg = Graph.atom_dgl_multigraph(
            atoms,
            cutoff=float(cutoff),
            max_neighbors=max_neighbors,
        )
        out = (
            model([g.to(device), lg.to(device)])
            .detach()
            .cpu()
            .numpy()
            .flatten()
            .tolist()[0]
        )
        fenp = (
            form_model([g.to(device), lg.to(device)])
            .detach()
            .cpu()
            .numpy()
            .flatten()
            .tolist()[0]
        )
        bg = (
            bg_model([g.to(device), lg.to(device)])
            .detach()
            .cpu()
            .numpy()
            .flatten()
            .tolist()[0]
        )
        if fenp < 0 and out > 5:
            print(
                "out",
                atoms.composition.reduced_formula,
                atoms.spacegroup(),
                out,
                fenp,
                bg,
            )
            print(atoms)
    except:
        pass
# get_multiple_predictions([Atoms.from_dict(i) for i in d],filename="pred_data.json")
