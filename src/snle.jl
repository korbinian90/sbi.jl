function snle(pθ, qψ, R, N)
    prθx0 = pθ
    D = []
    for r in 1:R
        for n in 1:N
            θn = sample(prθx0)
            xn = simulate(θn)
            push!(D, (θn, xn))
        end
        train!(qψ, D)
        prθx0 = qψ * pθ
    end
    return prθx0, qψ
end
