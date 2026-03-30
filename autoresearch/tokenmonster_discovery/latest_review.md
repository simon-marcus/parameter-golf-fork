# Review 0058

- keep: `False`
- headline: clean-v1 base regresses vs consistent-v1 at same resize 998
- next_direction: bracket the optimum around resize 998 on consistent-v1: try resize 990 and 1006 to find where improvement plateaus

Score 4.093 is worse than the frontier 4.074 at the same vocab size 998. The clean-v1 base with aggressive pruning produces higher dead_vocab_frac (3.8% vs 2.0%) and worse tokens_per_byte, indicating the pruning regime removes useful tokens. The consistent-v1 base remains clearly superior at this resize point.
