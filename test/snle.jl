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

@testset "sampling" begin
    using PyCall, Plots
    f(x) = logpdf(Product([Normal(), Normal()]), x)
    SliceSampler = pyimport("sbi.mcmc").SliceSampler
    posterior_sampler = SliceSampler([1.0, 1.0]; lp_f=f, verbose=true)
    s = posterior_sampler.gen(500)
    display(histogram(s))
end

@testset "custom multiround" begin
    using PyCall
    N = 5
    ndim = 3
    x0 = zeros(Float32, ndim)
    prior = Product([Uniform.(-2, 2) for _ in 1:ndim])
    simulator(θ) = θ .+ 1 .+ randn.(Float32) ./ 10
    qψ = sbi.build_maf(ones(Float32, N,ndim), ones(Float32, N,ndim))
    
    posterior = snle(simulator, prior, qψ, 1, N, x0)
    lp_f = θ -> sbi.log_prob(posterior, x0, θ)
    mcmc = pyimport("sbi.mcmc")
    pyimport("importlib")."reload"(mcmc)    
    posterior_sampler = mcmc.SliceSampler(x0; lp_f, verbose=true)
    samples = posterior_sampler.gen(10)
end

@testset "multiround" begin
    nrounds = 2
    N = 2
    ndim = 3
    x0 = zeros(Float32, ndim)
    prior = Product([Uniform.(-2, 2) for _ in 1:ndim])
    simulator(θ) = θ .+ 1 .+ randn.(Float32) ./ 10
    qψ = sbi.build_maf(ones(Float32, N,ndim), ones(Float32, N,ndim))
    snle(simulator, prior, qψ, nrounds, N, x0)
end
