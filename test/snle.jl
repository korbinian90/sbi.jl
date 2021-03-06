@testset "SNLE 1D" begin
    @show snle(1, 1, 1, 1)
    pθ = Uniform(-2, 2)
    @show snle(pθ, 1, 1, 1)
end

@testset "SNLE nD" begin
    ndim = 3
    pθ = Product([Uniform.(-2, 2) for _ in 1:ndim])
    @show snle(pθ, 1, 1, 1)
end

@testset "linear gaussian example" begin
    ndim = 3
    pθ = Product([Uniform.(-2, 2) for _ in 1:ndim])
    linear_gaussian(θ) = θ .+ 1.0 .+ 0.1 .* randn(size(θ))
    # prepare for sbi
end
