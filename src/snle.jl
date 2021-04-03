# from https://arxiv.org/abs/2101.04653, Appendix A4
function snle(simulator, prior, qψ=1, R=1, N=1, x0=nothing)
    if isnothing(x0) R=1 end
    posterior = prior
    D = Data()
    for r in 1:R
        θ = sample(posterior, x0, N)
        x = simulate(simulator, θ)
        push!(D, θ, x)
        train!(qψ, D)
        posterior = qψ * prior
    end
    return posterior
end

mutable struct Data
    θ
    x
end
Data() = Data(nothing, nothing)

function Base.push!(D::Data, θ, x)
    if isnothing(D.θ)
        D.θ = θ
        D.x = x
    else
        D.θ = vcat(D.θ, θ)
        D.x = vcat(D.x, x)
    end
end

struct NN
    maf
    N
end

function build_maf(x, y)
    py"""
    import torch
    from sbi.neural_nets.flow import build_maf

    def maf(batch_x, batch_y):
        return build_maf(batch_x=torch.tensor(batch_x), batch_y=torch.tensor(batch_y))
    """
    return NN(py"maf"(x, y), size(x,1))
end

function log_prob(p::NN, batch_x, by)
    function adaptsize(x)        
        if size(x,1) != p.N
            x = permutedims(x)
        end
        if size(x,1) == 1 < p.N
            x = repeat(x, p.N)
        end
        return x
    end
    batch_x = adaptsize(batch_x)
    by = adaptsize(by)
    m = p.maf
    py"""
    def log_prob(batch_x, by):
        t = torch.tensor(batch_x)
        c = torch.tensor(by)
        return $(m).log_prob(t, context=c)
    """
    return py"log_prob"(batch_x, by).detach().numpy()
end

function sample(p, x0, N)
    θ = permutedims(rand(p, N)) # Distributions.sampler? MCMC
    return Float32.(θ)
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

struct NeuralPosterior
    qψ
    p::Distribution
end

function log_prob(p::NeuralPosterior, x, θ)
    return Float32(first(sbi.log_prob(p.qψ, x, Float32.(θ)))) + logpdf(p.p, θ[:])
end

Base.:*(qψ::NN, p::Sampleable) = NeuralPosterior(qψ, p)

function sample(p::NeuralPosterior, x0, N)
    init = x0
    lp_f = θ -> log_prob(p, Float32.(x0), θ)
    SliceSampler = pyimport("sbi.mcmc").SliceSampler
    posterior_sampler = SliceSampler(init; lp_f=lp_f, verbose=true)
    return posterior_sampler.gen(N)
end
