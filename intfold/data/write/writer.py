from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from intfold.data.types import (
    Interface,
    Record,
    Structure,
)
from intfold.data.write.mmcif import to_mmcif
from intfold.data.write.pdb import to_pdb


def write_cif(structure, record, coord, plddts, output_path, output_format='mmcif'):
    # Compute chain map with masked removed, to be used later
    chain_map = {}
    for i, mask in enumerate(structure.mask):
        if mask:
            chain_map[len(chain_map)] = i
    # Remove masked chains completely
    structure = structure.remove_invalid_chains()
    # for model_idx in range(coord.shape[0]):
    # # Get model coord
    model_coord = coord.squeeze(0)
    model_plddts = plddts.squeeze(0)

    # New atom table
    atoms = structure.atoms
    # atoms["coords"] = coord_unpad
    atoms["coords"] = model_coord
    atoms["is_present"] = True

    # Mew residue table
    residues = structure.residues
    residues["is_present"] = True

    # Update the structure
    interfaces = np.array([], dtype=Interface)
    new_structure: Structure = replace(
        structure,
        atoms=atoms,
        residues=residues,
        interfaces=interfaces,
    )

    # Update chain info
    chain_info = []
    for chain in new_structure.chains:
        old_chain_idx = chain_map[chain["asym_id"]]
        old_chain_info = record.chains[old_chain_idx]
        new_chain_info = replace(
            old_chain_info,
            chain_id=int(chain["asym_id"]),
            valid=True,
        )
        chain_info.append(new_chain_info)

    # Save the structure
    if output_format == "pdb":
        with output_path.open("w") as f:
            f.write(to_pdb(new_structure, plddts=model_plddts))
    elif output_format == "mmcif":
        with output_path.open("w") as f:
            f.write(to_mmcif(new_structure, plddts=model_plddts))