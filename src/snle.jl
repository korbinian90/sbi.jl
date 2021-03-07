# from https://arxiv.org/abs/2101.04653, Appendix A4
function snle(simulator, prior, qψ=1, R=1, N=1)
    posterior = prior
    D = []
    for r in 1:R
        θ = sample(posterior, N)
        x = simulate(simulator, θ)
        push!(D, (θ, x))
        train!(qψ, D)
        posterior = qψ * prior
    end
    return posterior, qψ
end

function __initpy__()
    py"""
    import torch
    from sbi.neural_nets.flow import build_maf

    def maf(batch_x, batch_y):
        return build_maf(batch_x=torch.tensor(batch_x), batch_y=torch.tensor(batch_y))
    """
end

struct NN
    maf
end

function build_maf(x, y)
    return NN(py"maf"(x, y))
end

function sample(p, N)
    θ = rand(p, (N,)) # Distributions.sampler? MCMC
    return θ
end

function simulate(simulator, θ)
    x = simulator.(θ)
    return x
end

function train!(qψ, D)
end

function Base.:*(qψ, p::Sampleable)
    return p
end
