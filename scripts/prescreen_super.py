from jarvis.db.jsonutils import loadjson, dumpjson
from jarvis.core.atoms import Atoms
from alignn.pretrained import (
    get_multiple_predictions,
    get_prediction,
    get_figshare_model,
)
from jarvis.core.graphs import Graph
import torch
from alignn.models.alignn import ALIGNN, ALIGNNConfig

d = loadjson("eval_gen2d.json")
# d = loadjson("eval_gensup.json")
cutoff = 8
max_neighbors = 12
# filename = "/wrk/knc6/SuperConDeb/HT_DATA1053/Tc/out/checkpoint_200.pt"
filename = "checkpoint_200.pt"
output_features = 1
tc_model = ALIGNN(ALIGNNConfig(name="alignn", output_features=output_features))
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
device = "cpu"
tc_model.load_state_dict(torch.load(filename, map_location=device)["model"])
tc_model.to(device)
tc_model.eval()

form_model = get_figshare_model(
    model_name="jv_formation_energy_peratom_alignn"
)
form_model.to(device)


bg_model = get_figshare_model(model_name="jv_optb88vdw_bandgap_alignn")
bg_model.to(device)
print("size", len(d))
mem = []
for i in d:
    try:
        atoms = Atoms.from_dict(i)

        g, lg = Graph.atom_dgl_multigraph(
            atoms,
            cutoff=float(cutoff),
            max_neighbors=max_neighbors,
        )
        out = (
            tc_model([g.to(device), lg.to(device)])
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
        info = {}
        info["atoms"] = i
        info["pred"] = out
        info["fenp"] = fenp
        info["bg"] = bg
        info["formula"] = atoms.composition.reduced_formula
        mem.append(info)
        if fenp < 0 and out > 5:
            print(
                "formula,Spg,Tc,FormEnp,Bandgap",
                atoms.composition.reduced_formula,
                atoms.spacegroup(),
                out,
                fenp,
                bg,
            )
            print(atoms)
    except:
        pass
dumpjson(data=mem, filename="pred_data.json")
# get_multiple_predictions([Atoms.from_dict(i) for i in d],filename="pred_data.json")
