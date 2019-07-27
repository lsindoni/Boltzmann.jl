using Distributions


const DISTRIBUTIONS = [Degenerate, Gaussian, Bernoulli]

# The role of these tests is to ensure that the
# the general functionality always works and
# doesn't result in NaN or Inf weights.
# Not that the RBM performance is still good.
if :integration in TEST_GROUPS
    @testset "RBM Integration" begin
        n_vis = 100
        n_hid = 10

        for T in [Float32, Float64]
            rbm = RBM(T, Gaussian, Bernoulli, n_vis, n_hid)
            Boltzmann.test(rbm; debug=true)
        end
    end

    @testset "RBM getters" begin
        rbm = GRBM(4, 3)
        @test coef(rbm) == rbm.W'
        @test hbias(rbm) == rbm.hbias
        @test vbias(rbm) == rbm.vbias
    end

    @testset "RBM basic functionality" begin
        n_vis = 4
        n_hid = 3
        n_samples = 5
        rbm = GRBM(n_vis, n_hid)
        x = rand(n_vis, n_samples)
        fit(rbm, x; randomize=true)
        z = transform(rbm, x)
        @test size(z) == (n_hid, n_samples)
        x_new = generate(rbm, x; n_gibbs=2)
        @test size(x_new) == size(x)
        # TODO: systematically include and test all the options available in ctx
        fit(rbm, x; l1=true)
    end

    @testset "RBM sampling" begin
        n_vis = 4
        n_hid = 3
        n_samples = 5
        rbm = GRBM(n_vis, n_hid)
        z = Boltzmann.sample_hiddens(rbm, rand(n_vis, n_samples))
        @test size(z) == (n_hid, n_samples)
        x_new = Boltzmann.sample_visibles(rbm, z)
        @test size(x_new) == (n_vis, n_samples)
    end

end

if :benchmark in TEST_GROUPS
    @testset "RBM Benchmark" begin
        n_vis = 100
        n_hid = 10

        suite = BenchmarkGroup()

        for T in [Float32, Float64]
            rbm = RBM(T, Gaussian, Bernoulli, n_vis, n_hid)
            suite = benchmark!(rbm, suite; debug=true)
        end

        tune!(suite)
        results = run(suite, verbose=true, seconds=10)
    end
end
