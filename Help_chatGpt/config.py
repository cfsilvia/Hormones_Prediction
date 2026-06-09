from dataclasses import dataclass

@dataclass
class ArchetypeConfig:
    n_archetypes: int = 3
    n_pca_components: int = 3
    random_state: int = 42
    permutation_iterations: int = 300
