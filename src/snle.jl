# from DOI xxx, function xxx
function snle(pθ, qψ=1, R=1, N=1)
    prθx0 = pθ
    D = []
    for r in 1:R
        for n in 1:N
            θn = sample(prθx0) # Distributions.sampler?
            xn = simulate(θn)
            push!(D, (θn, xn))
        end
        train!(qψ, D)
        prθx0 = qψ * pθ
    end
    return prθx0, qψ
end

function sample(p)
    θ = rand(p)
    return θ
end

function simulate(θ)
    x = 1
    return x
end

function train!(qψ, D)
end

function Base.:*(qψ, p::Sampleable)
    return p
end
