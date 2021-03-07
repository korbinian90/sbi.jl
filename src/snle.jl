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
