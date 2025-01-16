import torch
import torch.nn as nn

def critic_loss(x_real, x_gen, critic, *, cr_real, cr_fake, _lambda=10, _lipschitz_factor=1):
    eps   = torch.rand_like(x_real)
    x_hat = torch.clone(eps * x_real + (1 - eps) * x_gen).requires_grad_(True)

    critic_v = critic(x_hat)

    grad_outputs = torch.ones_like(critic_v)
    x_hat_grads = torch.autograd.grad(
        outputs=critic_v,
        inputs=x_hat,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True
    )[0]

    gradient_penalty_loss = _lambda * torch.square(
        x_hat_grads.view(x_hat_grads.size(0), -1).norm(2, dim=1) - _lipschitz_factor)
    critic_loss = cr_fake - cr_real

    return torch.mean(critic_loss + gradient_penalty_loss)

def generator_loss(z, /, generator, critic):
    x_gen = generator(z)

    critic_v = critic(x_gen)

    return torch.mean(-critic_v)