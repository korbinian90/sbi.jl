@testset "PyCall maf test" begin    
    #sbi.__init__()
    try
        # always errors first time, probably wrong import order
        sbi.build_maf(ones(5), ones(2))
    catch
    end
    sbi.build_maf(ones(5), ones(2))
end

@testset "SNLE 1D" begin
    ndim = 1
    nbatch = 50
    pθ = Product([Uniform.(-2, 2) for _ in 1:ndim])
    simulator(θ) = 1 .+ θ .+ randn(eltype(θ), size(θ)) ./ 10
    qψ = sbi.build_maf(rand(Float32, nbatch, ndim), rand(Float32, nbatch, ndim))
    snle(simulator, pθ, qψ, 2, nbatch)
end

@testset "SNLE nD" begin
    ndim = 10
    nbatch = 50
    pθ = Product([Uniform.(-2, 2) for _ in 1:ndim])
    simulator(θ) = 1 .+ θ .+ randn(eltype(θ), size(θ)) ./ 10
    qψ = sbi.build_maf(rand(Float32, nbatch, ndim), rand(Float32, nbatch, ndim))
    snle(simulator, pθ, qψ, 2, nbatch)
end

@testset "Apply maf" begin
    ndim = 3
    nbatch = 50
    pθ = Product([Uniform.(-2, 2) for _ in 1:ndim])
    simulator(θ) = 1 .+ θ .+ randn(eltype(θ), size(θ)) ./ 10
    qψ = sbi.build_maf(rand(Float32, nbatch, ndim), rand(Float32, nbatch, ndim))
    snle(simulator, pθ, qψ, 2, nbatch)
end

@testset "test log_prob" begin
    N = 50
    qψ = sbi.build_maf(ones(N), ones(N))
    sbi.log_prob(qψ, ones(Float32, (N,1)), ones(Float32, (N,1)))    
end

@testset "test training" begin
    N = 50
    qψ = sbi.build_maf(ones(N), ones(N))
    D = sbi.Data(ones(N,1), ones(N,1))
    p1 = convert(Float64, collect(qψ.maf.parameters())[10][10])
    sbi.train!(qψ, D)
    p2 = convert(Float64, collect(qψ.maf.parameters())[10][10])
    @test p1 != p2
end
