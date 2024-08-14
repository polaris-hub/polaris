from fastpdb import struc


def zarr_to_pdb(atom_dict: dict):
    """Load a dictionary of arrays to fastpdb AtomArray"""

    atom_array = []

    # convert dictionary to array list of Atom object
    array_length = atom_dict["X"].shape[0]
    for ind in range(array_length):
        atom = struc.Atom(
            coord=(atom_dict["X"][ind], atom_dict["Y"][ind], atom_dict["Z"][ind]),
            chain_id=atom_dict["chain_id"][ind],
            res_id=atom_dict["res_id"][ind],
            ins_code=atom_dict["ins_code"][ind],
            res_name=atom_dict["res_name"][ind],
            hetero=atom_dict["hetero"][ind],
            atom_name=atom_dict["atom_name"][ind],
            element=atom_dict["element"][ind],
            b_factor=atom_dict["b_factor"][ind],
            occupancy=atom_dict["occupancy"][ind],
            charge=atom_dict["charge"][ind],
            atom_id=atom_dict["atom_id"][ind],
        )
        atom_array.append(atom)
    return struc.array(atom_array)
