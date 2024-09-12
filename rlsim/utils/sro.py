import numpy as np


def get_sro(atoms_list):
    nframe = len(atoms_list)

    species = list(sorted(set(atoms_list[0].get_atomic_numbers().tolist())))
    n_species = len(species)
    SRO = np.zeros((nframe, n_species, n_species))
    # info = {"species": species, "n_atoms": len(atoms_list[0])}

    for i in range(nframe):
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
