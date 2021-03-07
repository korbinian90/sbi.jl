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

@testset "PyCall maf test" begin    
    sbi.__initpy__()
    sbi.build_maf(ones(5), ones(2))
end

@testset "Apply maf" begin
    pθ = Uniform(-2, 2)
    simulator(θ) = 1.0 .+ θ .+ 0.1 .* randn(size(θ))
    nbatch_x, nbatch_y = 2, 2
    @show qψ = sbi.build_maf(ones(nbatch_x), ones(nbatch_y))
    @show snle(simulator, pθ, qψ, 2, 2)
end
