using Boltzmann
using HDF5

if :unit in TEST_GROUPS
    labels = ["foo", "bar"]
    rbms = [GRBM(4, 3), BernoulliRBM(3, 2)]
    test_net = DBN(rbms, labels)
    fields = [:W, :vbias, :hbias]

    @testset "RBM save and load" begin
        for (label, rbm) in zip(labels, rbms)
            fname = label * ".hdf5"
            h5open(fname, "w") do file
                save_params(file, rbm, label)
            end
            old = Dict(field => copy(getfield(rbm, field)) for field in fields)
            h5open(fname) do file
                load_params(file, rbm, label)
                for field in fields
                    @test getfield(rbm, field) == old[field]
                end
            end
            rm(fname)
        end
    end

    @testset "Net save and load" begin
        old_net = deepcopy(test_net)
        save_params("all_net.hdf5", test_net)
        load_params("all_net.hdf5", test_net)
        for i in 1:2
            for field in fields
                @test getfield(old_net[i], field) == getfield(test_net[i], field)
            end
        end
        rm("all_net.hdf5")
    end
end
