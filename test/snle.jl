@testset "SNLE 1D" begin
    pθ = Uniform(-2, 2)
    simulator(θ) = 1.0 .+ θ .+ 0.1 .* randn(size(θ))
    @show snle(simulator, pθ)
    @show snle(simulator, pθ, 1, 5, 5)
    # not finished
end

@testset "SNLE nD" begin
    ndim = 3
    pθ = Product([Uniform.(-2, 2) for _ in 1:ndim])
    simulator(θ) = 1.0 .+ θ .+ 0.1 .* randn(size(θ))
    @show snle(simulator, pθ)
    @show snle(simulator, pθ, 1, 5, 5)
    # not finished
end
