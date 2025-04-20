from typing import List

def combine_cu_seqlens(cu_seqlens_list: List[List[int]]) -> List[int]:
    """
    Combine cu_seqlens from multiple sequences.
    """
    cu_seqlens = []
    num_prev_tokens = 0
    for i in range(len(cu_seqlens_list)):
        for j in range(len(cu_seqlens_list[i])):
            if j == 0:
                continue
            cu_seqlens.append(cu_seqlens_list[i][j] + num_prev_tokens)
        # after the current item, update the num_prev_tokens
        num_prev_tokens = cu_seqlens[-1]

    cu_seqlens.insert(0, 0)
    return cu_seqlens
