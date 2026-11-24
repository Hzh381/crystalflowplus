"""按目标物性筛选生成结果。

读取 ``scripts/generation.py`` 写出的 ``eval_gen*.pt``，用 CrystalFlow+ 模型
的物性预测头给每个候选结构打分，保留 ``|Ê - E*|`` 最小的前 ``--top-k`` 个。

示例：
    python scripts/property_screen.py -m <model_path> -g <model_path>/eval_gen.pt \
        -k 0.0 --top-k 20 --write-cif
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from torch_geometric.data import Batch, Data

from eval_utils import get_crystals_list, get_scaler, load_model


def build_batch(gen):
    """把生成结果中的扁平数组重新组织成 PyG Batch。"""
    frac_coords = torch.as_tensor(gen["frac_coords"]).float()
    atom_types = torch.as_tensor(gen["atom_types"]).long()
    lengths = torch.as_tensor(gen["lengths"]).float()
    angles = torch.as_tensor(gen["angles"]).float()
    num_atoms = torch.as_tensor(gen["num_atoms"]).long()

    data_list = []
    offset = 0
    for i, n in enumerate(num_atoms.tolist()):
        data_list.append(
            Data(
                frac_coords=frac_coords[offset:offset + n],
                atom_types=atom_types[offset:offset + n],
                lengths=lengths[i].view(1, -1),
                angles=angles[i].view(1, -1),
                num_atoms=n,
                num_nodes=n,
            )
        )
        offset += n
    if offset != frac_coords.shape[0]:
        raise ValueError(
            f"atom count mismatch: num_atoms sums to {offset}, got {frac_coords.shape[0]} coordinates."
        )
    return Batch.from_data_list(data_list)


def to_structure(crystal):
    """把内部晶体字典转换为 pymatgen Structure；非法结构直接跳过。"""
    try:
        return Structure(
            lattice=Lattice.from_parameters(*crystal["lengths"].tolist(), *crystal["angles"].tolist()),
            species=crystal["atom_types"].tolist(),
            coords=crystal["frac_coords"],
            coords_are_cartesian=False,
        )
    except Exception as exc:  # 单个非法候选不应中断整个筛选流程
        print(f"skip malformed structure: {exc}")
        return None


def main(args):
    model_path = Path(args.model_path)
    model, _, cfg = load_model(model_path, load_data=False)
    if not getattr(model, "prop_enabled", False):
        raise SystemExit("该 checkpoint 没有物性预测头，请用 model=flow_plus 训练。")
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    prop_key = args.property or model.prop_key
    scaler = get_scaler(model, cfg, prop_key)
    target_scaled = float(scaler.transform([args.target])[0])

    gen = torch.load(args.gen_file, map_location="cpu")
    batch = build_batch(gen)
    batch = batch.to(next(model.parameters()).device)

    # 模型输出的是归一化空间中的物性，这里同时换算回原始单位用于展示。
    with torch.no_grad():
        pred_scaled = model.predict_property(batch).view(-1).float().cpu().numpy()
    pred_original = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).detach().cpu().numpy().reshape(-1)

    errors = np.abs(pred_scaled - target_scaled)
    order = np.argsort(errors)
    top_k = min(args.top_k, len(order))
    selected = order[:top_k]

    crystal_list = get_crystals_list(
        gen["frac_coords"], gen["atom_types"], gen["lengths"], gen["angles"], gen["num_atoms"]
    )
    selected_crystals = [crystal_list[i] for i in selected]

    out_path = Path(args.out) if args.out else Path(args.gen_file).with_name("screened.pt")
    torch.save(
        {
            "property": prop_key,
            "target": args.target,
            "target_scaled": target_scaled,
            "pred_original": pred_original,
            "pred_scaled": pred_scaled,
            "error": errors,
            "selected_indices": selected.tolist(),
            "crystal_list": selected_crystals,
        },
        out_path,
    )

    print(f"Scored {len(pred_original)} candidates for {prop_key} against target {args.target}.")
    print(f"Selected {top_k} -> {out_path}")
    for rank, i in enumerate(selected):
        print(f"  #{rank:02d} idx={i:5d} pred={pred_original[i]:+.4f} |err|={errors[i]:.4f} (scaled)")

    if args.write_cif:
        cif_dir = out_path.with_suffix("")
        cif_dir = cif_dir.parent / (cif_dir.name + "_cif")
        cif_dir.mkdir(parents=True, exist_ok=True)
        for rank, crystal in enumerate(selected_crystals):
            structure = to_structure(crystal)
            if structure is None:
                continue
            CifWriter(structure).write_file(str(cif_dir / f"{rank:03d}.cif"))
        print(f"CIFs written to {cif_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model_path", required=True, help="训练好的 CrystalFlow+ 模型目录")
    parser.add_argument("-g", "--gen_file", required=True, help="scripts/generation.py 生成的 eval_gen*.pt")
    parser.add_argument("-k", "--target", type=float, required=True, help="目标物性值（原始单位）")
    parser.add_argument("-p", "--property", default=None, help="物性键名（默认使用 model.prop_key）")
    parser.add_argument("--top-k", type=int, default=10, help="保留的候选数量")
    parser.add_argument("-o", "--out", default=None, help="输出 .pt 路径")
    parser.add_argument("--write-cif", action="store_true", help="同时把选中的结构写成 CIF")
    main(parser.parse_args())
