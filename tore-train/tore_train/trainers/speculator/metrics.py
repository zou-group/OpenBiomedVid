import torch
from transformers.logits_warper import RepetitionPenaltyLogitsProcessor, TopPLogitsWarper

def get_acceptance_rate(
    target_logits, draft_logits, temp=1.0, top_p=1.0, rep_penalty=1.0, input_ids=None
):
    # `target_logits` and `draft_logits` should have shape `(batch_size, seq_len, vocab_size)`.
    assert len(target_logits.shape) == 3
    assert len(draft_logits.shape) == 3
    draft_logits = draft_logits.to(target_logits.device)

    if rep_penalty != 1.0:
        processor = RepetitionPenaltyLogitsProcessor(rep_penalty)
        for i in range(1, target_logits.size(1)):
            processor(input_ids[:, :i], target_logits[:, i])
            processor(input_ids[:, :i], draft_logits[:, i])

    if top_p != 1.0:
        target_logits = TopPLogitsWarper(top_p)(input_ids, target_logits)
        draft_logits = TopPLogitsWarper(top_p)(input_ids, draft_logits)

    if temp != 1.0:
        target_logits = target_logits / temp
        draft_logits = draft_logits / temp

    target_p = torch.softmax(target_logits, dim=-1)
    draft_p = torch.softmax(draft_logits, dim=-1)

    acceptance_rate = (
        1 - (target_p - draft_p).abs().sum(dim=-1) / 2
    )  # `(batch_size, seq_len)`

    assert len(acceptance_rate.shape) == 2

    return acceptance_rate
