import torch
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For regularized Bellman operator - direct use of analytic policy?
# Let's go with Mellowmax...?
def mellowmax_policy(q, lamb=0.2, tau=0):
    # TODO: maximizer for regularized Bellman operator
    mu = (1 / q.data.size(1)) * torch.ones(q.data.shape).to(device) # Random policy
    c = mellowmax_maximum(q.data, lamb=lamb, tau=tau)
    log_policy = q.data / (lamb + tau) + ( lamb / (lamb + tau) ) * mu.log() - c / (lamb + tau)
    policy = log_policy.clamp(-10.0, 10.0).exp() # At policy, the values explode largely - need clamping the values!
    # print("normalized well? : ", policy.sum(1))
    # print("policy : ", policy / policy.sum(1))
    m = Categorical(policy / policy.sum())
    return m.sample()


def mellowmax_maximum(q, lamb=0.2, tau=0):
    # TODO: maximum value for regularized Bellman operator
    # print("q: ", q.shape)
    v = q.mean(dim=1, keepdim=True)
    # print("v: ", v.shape)
    expsum = ( (q - v.expand(-1, q.size(1))) / (lamb + tau) ).exp().to(device)
    mu = (1 / expsum.size(1)) * torch.ones(expsum.shape).to(device)
    logsumexp = (expsum * (mu ** (lamb / (lamb + tau)))).sum(axis=1, keepdim=True).log() + v / (lamb + tau)
    # print("logsumexp: ", logsumexp)
    return (lamb + tau) * logsumexp.view(-1)


def soft_maximum(q, mu, lamb=0.2, tau=0):
    # TODO: maximum value for regularized Bellman operator
    # print("q: ", q.shape)
    v = q.mean(dim=1, keepdim=True) # For logsumexp trick
    # print("v: ", v.shape)
    expsum = ( (q - v.expand(-1, q.size(1))) / (lamb + tau) ).exp().to(device)
    # print("mu: ", mu.shape)
    logsumexp = (expsum * (mu ** (lamb / (lamb + tau)))).sum(axis=1, keepdim=True).log() + v / (lamb + tau)
    # print("logsumexp: ", logsumexp)
    return (lamb + tau) * logsumexp.view(-1)