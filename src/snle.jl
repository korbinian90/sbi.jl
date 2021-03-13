# from https://arxiv.org/abs/2101.04653, Appendix A4
function snle(simulator, prior, qψ=1, R=1, N=1)
    posterior = prior
    D = Data(nothing, nothing)
    for r in 1:R
        θ = sample(posterior, N)
        x = simulate(simulator, θ)
        push!(D, θ, x)
        train!(qψ, D)
        posterior = qψ * prior
    end
    return posterior, qψ
end

mutable struct Data
    θ
    x
end

function Base.push!(D::Data, θ, x)
    if isnothing(D.θ)
        D.θ = θ
        D.x = x
    else
        append!(D.θ, θ)
        append!(D.x, x)
    end
end

struct NN
    maf
end

function build_maf(x, y)
    py"""
    import torch
    from sbi.neural_nets.flow import build_maf

    def maf(batch_x, batch_y):
        return build_maf(batch_x=torch.tensor(batch_x), batch_y=torch.tensor(batch_y))
    """
    return NN(py"maf"(x, y))
end

function log_prob(p::NN, batch_x, by)
    m = p.maf
    py"""
    def log_prob(batch_x, by):
        t = torch.tensor(batch_x)
        c = torch.tensor(by)
        return $(m).log_prob(t, context=c)
    """
    return py"log_prob"(batch_x, by)
end

function sample(p, N)
    θ = rand(p, (N,)) # Distributions.sampler? MCMC
    return map(x -> Float32.(x), θ)
end

function simulate(simulator, θ)
    x = simulator.(θ)
    return x
end

function train!(qψ, D)
    torch = pyimport("torch")
    data = pyimport("torch.utils").data
    formatdata(d) = torch.Tensor(permutedims(hcat(d...)))
    dataset = data.TensorDataset(formatdata(D.θ), formatdata(D.x))
    train_loader = data.DataLoader(dataset; batch_size=50, drop_last=true)

    optimizer = torch.optim.Adam(qψ.maf.parameters())
    qψ.maf.train()
    for batch in train_loader
        optimizer.zero_grad()
        log_prob = qψ.maf.log_prob(batch[1], context=batch[2])
        loss = -torch.mean(log_prob)
        loss.backward()
        optimizer.step()
    end
end

function Base.:*(qψ, p::Sampleable)
    return p
end
