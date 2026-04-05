import numpy as np
from tqdm import tqdm
from ase import Atoms

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


def get_binary_sro(atoms_list: list[Atoms]) -> np.ndarray:
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

    return alpha