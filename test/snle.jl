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

@testset "tests" begin
    using PyCall
    N = 50
    pθ = Uniform(-2, 2)
    simulator(θ) = 1.0 .+ θ .+ 0.1 .* randn(size(θ))
    nbatch_x, nbatch_y = N, N
    qψ = sbi.build_maf(ones(nbatch_x), ones(nbatch_y))
   
    sbi.log_prob(qψ, ones(Float32, (N,1)), ones(Float32, (N,1)))

    D = sbi.Data(ones(nbatch_x,1), ones(nbatch_y,1))
    torch = pyimport("torch")
    data = pyimport("torch.utils").data
    dataset = data.TensorDataset(torch.Tensor(D.θ), torch.Tensor(D.x))
    train_loader = data.DataLoader(dataset; batch_size=50, drop_last=true)
    optimizer = torch.optim.Adam(qψ.maf.parameters())
    for batch in train_loader
        optimizer.zero_grad()
        log_prob = qψ.maf.log_prob(batch[1], context=batch[2])
        loss = -torch.mean(log_prob)
        loss.backward()
        optimizer.step()
    end
end
