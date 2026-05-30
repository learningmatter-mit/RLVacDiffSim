import numpy as np
from tqdm import tqdm
from ase import Atoms
from ase.build import bulk
from ase.geometry import get_distances

def get_sro(atoms_list: list[Atoms]) -> np.ndarray:
    nframe = len(atoms_list)

    species = list(sorted(set(atoms_list[0].get_atomic_numbers().tolist())))
    n_species = len(species)
    SRO = np.zeros((nframe, n_species, n_species))
    # info = {"species": species, "n_atoms": len(atoms_list[0])}

    for i in tqdm(range(nframe), desc="Processing trajectories"):
        atoms = atoms_list[i]
        # n_atoms = len(atoms)
        dist_all = atoms.get_all_distances(mic=True)
        dist_list = dist_all.flatten()
        dist_list = dist_list[dist_list > 0.1]
        NN_dist = np.min(dist_list)
        r_cut = (1 + np.sqrt(2)) / 2 * NN_dist

        pairs = (dist_all > 0.1) * (dist_all < r_cut)
        total_pairs = np.sum(pairs) / 2
        atomic_numbers = atoms.get_atomic_numbers()
        specie_list = [atomic_numbers == species[n] for n in range(n_species)]
        n_specie_list = [np.sum(specie_list[n]) for n in range(n_species)]
        n_specie_list = np.array(n_specie_list) / np.sum(n_specie_list)

        for n1 in range(n_species):
            for n2 in range(n1 + 1):
                number_of_pairs = (
                    np.sum(pairs * specie_list[n1][None, :] * specie_list[n2][:, None])
                    / 2
                )
                SRO[i, n1, n2] = (
                    1
                    - (number_of_pairs / total_pairs)
                    / n_specie_list[n1]
                    / n_specie_list[n2]
                )

                if n1 != n2:
                    SRO[i, n2, n1] = SRO[i, n1, n2]

    return SRO

def get_sro_from_atoms(atoms) -> np.ndarray:
    species = list(sorted(set(atoms.get_atomic_numbers().tolist())))
    n_species = len(species)
    SRO = np.zeros((n_species, n_species))
    # n_atoms = len(atoms)
    dist_all = atoms.get_all_distances(mic=True)
    dist_list = dist_all.flatten()
    dist_list = dist_list[dist_list > 0.1]
    NN_dist = np.min(dist_list)
    r_cut = (1 + np.sqrt(2)) / 2 * NN_dist

    pairs = (dist_all > 0.1) * (dist_all < r_cut)
    total_pairs = np.sum(pairs) / 2
    atomic_numbers = atoms.get_atomic_numbers()
    specie_list = [atomic_numbers == species[n] for n in range(n_species)]
    n_specie_list = [np.sum(specie_list[n]) for n in range(n_species)]
    n_specie_list = np.array(n_specie_list) / np.sum(n_specie_list)

    for n1 in range(n_species):
        for n2 in range(n1 + 1):
            number_of_pairs = (
                np.sum(pairs * specie_list[n1][None, :] * specie_list[n2][:, None])
                / 2
            )
            SRO[n1, n2] = (
                1
                - (number_of_pairs / total_pairs)
                / n_specie_list[n1]
                / n_specie_list[n2]
            )

            if n1 != n2:
                SRO[n2, n1] = SRO[n1, n2]

    return SRO


def get_binary_sro_from_atoms(atoms) -> float:
    """
    Compute the Warren-Cowley SRO parameter for a binary alloy.

    For species A and B:
        α_AB = 1 - P(B|A) / c_B
    where P(B|A) is the conditional probability of finding a B neighbor
    given an A atom, and c_B is the bulk concentration of B.

    Returns a single scalar:
        α_AB > 0  → like-atom pairs preferred  (ordering tendency absent)
        α_AB < 0  → unlike-atom pairs preferred (short-range order)
        α_AB = 0  → random solid solution
    """
    atomic_numbers = atoms.get_atomic_numbers()
    species = sorted(set(atomic_numbers.tolist()))
    assert len(species) == 2, "This function is for binary systems only."

    # --- nearest-neighbour cutoff (FCC convention) ---
    dist_all = atoms.get_all_distances(mic=True)
    dist_flat = dist_all.flatten()
    dist_flat = dist_flat[dist_flat > 0.1]
    NN_dist = np.min(dist_flat)
    r_cut = (1 + np.sqrt(2)) / 2 * NN_dist

    pairs = (dist_all > 0.1) & (dist_all < r_cut)
    total_pairs = np.sum(pairs) / 2

    mask_A = atomic_numbers == species[0]
    mask_B = atomic_numbers == species[1]

    c_A = np.sum(mask_A) / len(atomic_numbers)
    c_B = np.sum(mask_B) / len(atomic_numbers)

    # number of A-B pairs (count each bond once)
    n_AB = np.sum(pairs * mask_A[None, :] * mask_B[:, None]) / 2

    # Warren-Cowley α_AB (mirrors off-diagonal element of get_sro_from_atoms)
    alpha_AB = 1.0 - (n_AB / total_pairs) / (c_A * c_B)
    return alpha_AB


def get_binary_lro_from_atoms(atoms, ref_atoms=None) -> float:
    """
    Compute the long-range order (LRO) parameter for a binary FCC alloy (L1_2 phase by default).

    If ref_atoms is provided, the function computes the LRO parameter relative to the
    reference structure. This general Bragg-Williams LRO parameter is defined as:
        η = (r - c) / (1 - ν)
    where:
        - c is the overall concentration of the minority species.
        - ν is the fraction of sites occupied by the minority species in the perfect reference structure.
        - r is the fraction of correct reference sites occupied by the minority species.

    If ref_atoms is None, it defaults to the L1_2 phase where:
        - ν = 0.25 is the fraction of sites belonging to the minority sublattice.
        - r is the fraction of minority sublattice sites occupied by the minority species.
    """
    atomic_numbers = atoms.get_atomic_numbers()
    species = sorted(set(atomic_numbers.tolist()))
    assert len(species) == 2, "This function is for binary systems only."

    # Identify the minority species in the current structure
    c_A = np.sum(atomic_numbers == species[0]) / len(atomic_numbers)
    c_B = np.sum(atomic_numbers == species[1]) / len(atomic_numbers)
    if c_A < c_B:
        minority_specie = species[0]
        c_min = c_A
    else:
        minority_specie = species[1]
        c_min = c_B

    cell = atoms.get_cell()
    pbc = atoms.get_pbc()
    atom_pos = atoms.get_positions()

    if ref_atoms is not None:
        ref_atomic_numbers = ref_atoms.get_atomic_numbers()
        ref_species = sorted(set(ref_atomic_numbers.tolist()))
        assert len(ref_species) == 2, "Reference structure must be binary."

        # Identify minority species in reference structure to define the target sublattice
        c_ref_A = np.sum(ref_atomic_numbers == ref_species[0]) / len(ref_atomic_numbers)
        c_ref_B = np.sum(ref_atomic_numbers == ref_species[1]) / len(ref_atomic_numbers)
        if c_ref_A < c_ref_B:
            ref_minority_specie = ref_species[0]
            nu = c_ref_A
        else:
            ref_minority_specie = ref_species[1]
            nu = c_ref_B

        # Map actual atoms to reference sites
        ref_sites = ref_atoms.get_positions()
        _, dists = get_distances(atom_pos, ref_sites, cell=cell, pbc=pbc)
        atom_to_site = np.argmin(dists, axis=1)

        # Correct sites for reference minority species
        correct_sites_mask = (ref_atomic_numbers == ref_minority_specie)
        
        # Check current minority species occupancy on those sites
        curr_minority_mask = (atomic_numbers == ref_minority_specie)
        mapped_sites = atom_to_site[curr_minority_mask]
        
        correct_count = np.sum(correct_sites_mask[mapped_sites])
        total_correct_sites = np.sum(correct_sites_mask)
        
        r = correct_count / total_correct_sites
        c_min_ref = np.sum(curr_minority_mask) / len(atomic_numbers)
        
        eta = (r - c_min_ref) / (1.0 - nu)
        return max(0.0, float(eta))

    # Default to L1_2 phase (nu = 0.25)
    # Estimate nearest-neighbour distance and lattice parameter
    dist_all = atoms.get_all_distances(mic=True)
    dist_flat = dist_all.flatten()
    dist_flat = dist_flat[dist_flat > 0.1]
    NN_dist = np.min(dist_flat)
    lattice_parameter = np.sqrt(2) * NN_dist

    # Build perfect FCC reference lattice of the same dimensions
    Nrepeat = np.round(cell.diagonal() / lattice_parameter).astype(int)
    perfect = bulk("Cu", 'fcc', a=lattice_parameter, cubic=True).repeat(Nrepeat)
    lattice_sites = perfect.get_positions()

    # Map actual atoms to the nearest perfect lattice sites
    _, dists = get_distances(atom_pos, lattice_sites, cell=cell, pbc=pbc)
    atom_to_site = np.argmin(dists, axis=1)

    # Determine sublattice for each perfect lattice site
    site_keys = np.round(lattice_sites / lattice_parameter * 2).astype(int) % 2
    raw_indices = np.sum(site_keys * [1, 2, 4], axis=1)
    unique_vals = sorted(np.unique(raw_indices))
    
    if len(unique_vals) != 4:
        raise ValueError(f"Structure is not a valid FCC supercell (expected 4 sublattices, found {len(unique_vals)}).")
    
    val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
    site_sublattices = np.array([val_to_idx[val] for val in raw_indices])

    # Find which perfect sites are occupied by the minority species
    minority_mask = (atomic_numbers == minority_specie)
    minority_sites = atom_to_site[minority_mask]

    # Calculate occupancy of minority species on each sublattice
    sites_per_sublattice = len(perfect) // 4
    sublattice_counts = np.zeros(4, dtype=int)
    for site_idx in minority_sites:
        subl = site_sublattices[site_idx]
        sublattice_counts[subl] += 1

    # The minority sublattice is the one with the maximum occupancy of the minority species
    r = np.max(sublattice_counts) / sites_per_sublattice

    # Bragg-Williams LRO parameter
    nu = 0.25
    eta = (r - c_min) / (1.0 - nu)
    return max(0.0, float(eta))


def is_l12_phase(atoms, tol: float = 1e-2) -> bool:
    """
    Determine if a binary FCC structure is in the L1_2 phase.

    A perfect L1_2 phase (A3B or AB3 in an FCC lattice) is characterized by:
    1. A stoichiometry of approximately 3:1 (or 1:3).
    2. A Warren-Cowley SRO parameter α_AB ≈ -1/3.
       This corresponds to all minority atoms being completely surrounded by
       majority atoms in their first nearest-neighbor shell.

    Args:
        atoms: ASE Atoms object.
        tol: Tolerance for the stoichiometry and SRO parameter.

    Returns:
        bool: True if the structure is considered to be L1_2, False otherwise.
    """
    atomic_numbers = atoms.get_atomic_numbers()
    species = sorted(set(atomic_numbers.tolist()))
    if len(species) != 2:
        return False

    c_A = np.sum(atomic_numbers == species[0]) / len(atomic_numbers)
    c_B = np.sum(atomic_numbers == species[1]) / len(atomic_numbers)

    # Check for approximately 3:1 or 1:3 stoichiometry
    if not (np.isclose(c_A, 0.75, atol=tol) or np.isclose(c_B, 0.75, atol=tol)):
        return False

    alpha_AB = get_binary_sro_from_atoms(atoms)
    
    # Perfect L1_2 has alpha_AB = -1/3
    return bool(np.isclose(alpha_AB, -1/3, atol=tol))


def get_binary_sro(atoms_list: list[Atoms], return_l12_phase: bool = False, tol: float = 1e-2) -> np.ndarray | tuple[np.ndarray, list[bool]]:
    """
    Compute the Warren-Cowley SRO parameter for each frame in a binary-alloy
    trajectory.

    Returns:
        alpha : np.ndarray of shape (nframe,)
            α_AB for every frame.
            α_AB > 0  → like-atom pairs preferred  (ordering tendency absent)
            α_AB < 0  → unlike-atom pairs preferred (short-range order)
            α_AB = 0  → random solid solution
    """
    species = sorted(set(atoms_list[0].get_atomic_numbers().tolist()))
    assert len(species) == 2, "This function is for binary systems only."

    nframe = len(atoms_list)
    alpha = np.zeros(nframe)
    if return_l12_phase:
        l12_phase_list = []

    for i in tqdm(range(nframe), desc="Processing trajectories"):
        atoms = atoms_list[i]
        atomic_numbers = atoms.get_atomic_numbers()

        # --- nearest-neighbour cutoff (FCC convention) ---
        dist_all = atoms.get_all_distances(mic=True)
        dist_flat = dist_all.flatten()
        dist_flat = dist_flat[dist_flat > 0.1]
        NN_dist = np.min(dist_flat)
        r_cut = (1 + np.sqrt(2)) / 2 * NN_dist

        pairs = (dist_all > 0.1) & (dist_all < r_cut)
        total_pairs = np.sum(pairs) / 2

        mask_A = atomic_numbers == species[0]
        mask_B = atomic_numbers == species[1]

        c_A = np.sum(mask_A) / len(atomic_numbers)
        c_B = np.sum(mask_B) / len(atomic_numbers)

        n_AB = np.sum(pairs * mask_A[None, :] * mask_B[:, None]) / 2

        alpha[i] = 1.0 - (n_AB / total_pairs) / (c_A * c_B)

        if return_l12_phase:
            l12_phase_list.append(is_l12_phase(atoms, tol=tol))

    if return_l12_phase:
        return alpha, l12_phase_list
    else:
        return alpha


def get_binary_lro(atoms_list: list[Atoms], ref_atoms=None) -> np.ndarray:
    """
    Compute the long-range order (LRO) parameter for each frame in a binary-alloy
    trajectory.

    If ref_atoms is provided, the LRO parameter is computed relative to the given
    perfectly ordered reference structure.
    """
    nframe = len(atoms_list)
    lro = np.zeros(nframe)

    if nframe == 0:
        return lro

    if ref_atoms is not None:
        ref_atomic_numbers = ref_atoms.get_atomic_numbers()
        ref_species = sorted(set(ref_atomic_numbers.tolist()))
        assert len(ref_species) == 2, "Reference structure must be binary."

        c_ref_A = np.sum(ref_atomic_numbers == ref_species[0]) / len(ref_atomic_numbers)
        c_ref_B = np.sum(ref_atomic_numbers == ref_species[1]) / len(ref_atomic_numbers)
        if c_ref_A < c_ref_B:
            ref_minority_specie = ref_species[0]
            nu = c_ref_A
        else:
            ref_minority_specie = ref_species[1]
            nu = c_ref_B

        ref_sites = ref_atoms.get_positions()
        total_correct_sites = np.sum(ref_atomic_numbers == ref_minority_specie)
        correct_sites_mask = (ref_atomic_numbers == ref_minority_specie)

        for i in tqdm(range(nframe), desc="Processing LRO trajectories"):
            atoms = atoms_list[i]
            curr_atomic_numbers = atoms.get_atomic_numbers()
            curr_cell = atoms.get_cell()
            curr_pbc = atoms.get_pbc()
            curr_pos = atoms.get_positions()

            _, dists = get_distances(curr_pos, ref_sites, cell=curr_cell, pbc=curr_pbc)
            atom_to_site = np.argmin(dists, axis=1)

            curr_minority_mask = (curr_atomic_numbers == ref_minority_specie)
            mapped_sites = atom_to_site[curr_minority_mask]

            correct_count = np.sum(correct_sites_mask[mapped_sites])
            r = correct_count / total_correct_sites
            c_min_ref = np.sum(curr_minority_mask) / len(curr_atomic_numbers)

            eta = (r - c_min_ref) / (1.0 - nu)
            lro[i] = max(0.0, float(eta))

        return lro

    # Default to L1_2 phase (nu = 0.25)
    # Initialize reference lattice using the first frame
    ref_atoms_init = atoms_list[0]
    atomic_numbers = ref_atoms_init.get_atomic_numbers()
    species = sorted(set(atomic_numbers.tolist()))
    assert len(species) == 2, "This function is for binary systems only."

    c_A = np.sum(atomic_numbers == species[0]) / len(atomic_numbers)
    c_B = np.sum(atomic_numbers == species[1]) / len(atomic_numbers)
    if c_A < c_B:
        minority_specie = species[0]
    else:
        minority_specie = species[1]

    cell = ref_atoms_init.get_cell()

    # Estimate nearest-neighbour distance and lattice parameter
    dist_all = ref_atoms_init.get_all_distances(mic=True)
    dist_flat = dist_all.flatten()
    dist_flat = dist_flat[dist_flat > 0.1]
    NN_dist = np.min(dist_flat)
    lattice_parameter = np.sqrt(2) * NN_dist

    # Build perfect FCC reference lattice
    Nrepeat = np.round(cell.diagonal() / lattice_parameter).astype(int)
    perfect = bulk("Cu", 'fcc', a=lattice_parameter, cubic=True).repeat(Nrepeat)
    lattice_sites = perfect.get_positions()
    sites_per_sublattice = len(perfect) // 4

    # Determine sublattice for each perfect lattice site
    site_keys = np.round(lattice_sites / lattice_parameter * 2).astype(int) % 2
    raw_indices = np.sum(site_keys * [1, 2, 4], axis=1)
    unique_vals = sorted(np.unique(raw_indices))
    if len(unique_vals) != 4:
        raise ValueError(f"Structure is not a valid FCC supercell (expected 4 sublattices, found {len(unique_vals)}).")
    val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
    site_sublattices = np.array([val_to_idx[val] for val in raw_indices])

    nu = 0.25

    for i in tqdm(range(nframe), desc="Processing LRO trajectories"):
        atoms = atoms_list[i]
        curr_atomic_numbers = atoms.get_atomic_numbers()
        curr_cell = atoms.get_cell()
        curr_pbc = atoms.get_pbc()
        curr_pos = atoms.get_positions()

        # Check if cell has changed (e.g. in NPT simulations)
        if not np.allclose(curr_cell, cell):
            cell = curr_cell
            # Recalculate NN_dist and lattice_parameter
            dist_all = atoms.get_all_distances(mic=True)
            dist_flat = dist_all.flatten()
            dist_flat = dist_flat[dist_flat > 0.1]
            NN_dist = np.min(dist_flat)
            lattice_parameter = np.sqrt(2) * NN_dist
            
            # Regenerate perfect lattice and sublattices
            Nrepeat = np.round(cell.diagonal() / lattice_parameter).astype(int)
            perfect = bulk("Cu", 'fcc', a=lattice_parameter, cubic=True).repeat(Nrepeat)
            lattice_sites = perfect.get_positions()
            sites_per_sublattice = len(perfect) // 4
            
            site_keys = np.round(lattice_sites / lattice_parameter * 2).astype(int) % 2
            raw_indices = np.sum(site_keys * [1, 2, 4], axis=1)
            unique_vals = sorted(np.unique(raw_indices))
            if len(unique_vals) != 4:
                raise ValueError(f"Structure is not a valid FCC supercell (expected 4 sublattices, found {len(unique_vals)}).")
            val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
            site_sublattices = np.array([val_to_idx[val] for val in raw_indices])

        # Overall concentration of the minority species in the current frame
        c_min = np.sum(curr_atomic_numbers == minority_specie) / len(curr_atomic_numbers)

        # Map actual atoms to reference lattice sites
        _, dists = get_distances(curr_pos, lattice_sites, cell=curr_cell, pbc=curr_pbc)
        atom_to_site = np.argmin(dists, axis=1)

        # Count minority atoms on each sublattice
        minority_mask = (curr_atomic_numbers == minority_specie)
        minority_sites = atom_to_site[minority_mask]

        sublattice_counts = np.zeros(4, dtype=int)
        for site_idx in minority_sites:
            subl = site_sublattices[site_idx]
            sublattice_counts[subl] += 1

        r = np.max(sublattice_counts) / sites_per_sublattice
        eta = (r - c_min) / (1.0 - nu)
        lro[i] = max(0.0, float(eta))

    return lro