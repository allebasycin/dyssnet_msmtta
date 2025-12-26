import torch

class MultiScaleMixupTTA:
    """Ablation: Multi-scale patterns with fixed lambda per scale"""

    def __init__(
            self,
            num_local_pairs: int = 2,
            num_medium_pairs: int = 2,
            num_global_pairs: int = 3,
            seed: int = 42
    ):
        self.num_local_pairs = num_local_pairs
        self.num_medium_pairs = num_medium_pairs
        self.num_global_pairs = num_global_pairs
        self.seed = seed
        self.last_mixing_info = []

    def get_calibrated_lambda(self, scale, outputs_i, outputs_j, base_diversity):
        """Fixed lambda per scale"""
        if scale == 'local':
            return 0.30  # Conservative
        elif scale == 'medium':
            return 0.50  # Moderate
        else:  # global
            return 0.70  # Aggressive

    def _get_deterministic_diversity_score(self, idx1, idx2, N_aug):
        """Same as original"""
        distance = abs(idx2 - idx1)
        max_distance = N_aug - 1
        return distance / max_distance

    def select_deterministic_patterns(self, N_aug):
        """Use ORIGINAL pattern selection logic"""
        patterns = []

        # Local: evenly-spaced adjacent pairs
        local_step = max(1, N_aug // (self.num_local_pairs + 1))
        for k in range(self.num_local_pairs):
            i = k * local_step
            j = min(i + 1, N_aug - 1)
            if i < j:
                diversity = self._get_deterministic_diversity_score(i, j, N_aug)
                patterns.append((i, j, 'local', diversity))

        # Medium: quarter-distance pairs
        medium_step = max(1, N_aug // 4)
        for k in range(self.num_medium_pairs):
            i = k * medium_step
            j = min(i + medium_step, N_aug - 1)
            if i < j:
                diversity = self._get_deterministic_diversity_score(i, j, N_aug)
                patterns.append((i, j, 'medium', diversity))

        # Global: extreme pairs
        global_pairs = [
            (0, N_aug - 1),
            (0, N_aug // 2),
            (N_aug // 2, N_aug - 1),
        ]
        for i, j in global_pairs[:self.num_global_pairs]:
            if i < j:
                diversity = self._get_deterministic_diversity_score(i, j, N_aug)
                patterns.append((i, j, 'global', diversity))

        return patterns

    def apply_deterministic_mixup(self, all_outputs):
        """Apply mixup with fixed lambdas"""
        N_aug = all_outputs.shape[0]

        if N_aug < 2:
            return all_outputs

        mixing_patterns = self.select_deterministic_patterns(N_aug)

        mixed_outputs = []
        mixing_info = []

        for idx1, idx2, scale, base_diversity in mixing_patterns:
            if idx1 < N_aug and idx2 < N_aug:
                # Get fixed lambda (no confidence needed)
                lambda_val = self.get_calibrated_lambda(
                    scale, None, None, base_diversity
                )

                # Apply mixup
                mixed = lambda_val * all_outputs[idx1] + \
                        (1 - lambda_val) * all_outputs[idx2]
                mixed_outputs.append(mixed)

                mixing_info.append({
                    'idx1': idx1,
                    'idx2': idx2,
                    'scale': scale,
                    'diversity': base_diversity,
                    'lambda': lambda_val
                })

        if mixed_outputs:
            mixed_outputs = torch.stack(mixed_outputs, dim=0)
            all_outputs = torch.cat([all_outputs, mixed_outputs], dim=0)

        self.last_mixing_info = mixing_info

        return all_outputs
