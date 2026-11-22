import numpy as np

def dpo_loss(log_probs_chosen_policy: list, log_probs_rejected_policy: list,
            log_probs_chosen_ref: list, log_probs_rejected_ref: list,
            beta: float) -> dict:
    """
    Compute the Direct Preference Optimization (DPO) loss.
    
    Args:
        log_probs_chosen_policy: Log-probs of chosen responses under policy
        log_probs_rejected_policy: Log-probs of rejected responses under policy
        log_probs_chosen_ref: Log-probs of chosen responses under reference model
        log_probs_rejected_ref: Log-probs of rejected responses under reference model
        beta: Temperature parameter for KL constraint strength
    
    Returns:
        Dictionary with 'loss', 'chosen_rewards', and 'rejected_rewards'
    """
    # Your code here
    chosen_diff = np.array(log_probs_chosen_policy) - np.array(log_probs_chosen_ref)
    rej_diff = np.array(log_probs_rejected_policy) - np.array(log_probs_rejected_ref)

    # -log(sigmoid(actual KL))
    loss = np.mean(-np.log(sigmoid(beta*(chosen_diff - rej_diff))))

    chosen_rewards = [round(i, 4) for i in (beta*chosen_diff)]
    rejected_rewards = [round(i, 4) for i in (beta*rej_diff)]
    
    return {"loss": float(round(loss, 4)), "chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}

def sigmoid(z):
  return 1/(1 + np.exp(-z))
