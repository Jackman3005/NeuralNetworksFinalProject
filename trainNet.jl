#This file initializes the Mocha settings and then trains a network
#Use "julia trainNet.jl --help" for help with CLA's

#For parsing CLA's
require("ArgParse")
using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    #Setting up the available CLA's
    @add_arg_table s begin
        "--numBLASThreads", "-B"
            help = "The number of BLAS threads to use"
            arg_type = Int
            default = 1

        "--numOMPThreads", "-O"
            help = "another option with an argument"
            arg_type = Int
            default = 1

        "--useCUDA", "-C"
            help = "A flag for whether CUDA should be used"
            action = :store_true

        "--dontUseNativeExt", "-N"
            help = "A flag for whether c++ Native extension should be used, only if not using GPU"
            action = :store_true

        "--dataFile", "-F"
            help = "path to the data file you want to use"
            arg_type = String
            default = ""

        "--basePath", "-P"
            help = "path to the folder you want to write to"
            arg_type = String
            default = pwd()
    end

    return parse_args(s)
end

# Parse CLA's and do system setup
parsed_args = parse_commandline()
println(parsed_args)
if parsed_args["useCUDA"]
  #We are going to use CUDA
  ENV["MOCHA_USE_CUDA"] = "true"
else
  #Don't use GPU
  if !parsed_args["dontUseNativeExt"]
    #this won't run by default
    ENV["MOCHA_USE_NATIVE_EXT"] = "true"
  end
  #
  ENV["OMP_NUM_THREADS"] = parsed_args["numOMPThreads"]
  blas_set_num_threads(parsed_args["numBLASThreads"])
end

data_path = parsed_args["dataFile"]
base_path = parsed_args["basePath"]

using Mocha

#####################################################
#####             Begin Netowrk                ######
#####################################################

# fix the random seed to make results reproducable
srand(12345678)

data_layer  = HDF5DataLayer(name="train-data", source=data_path, batch_size=100)
# each fully connected layer uses a ReLU activation and a constraint on the L2 norm of the weights
fc1_layer   = InnerProductLayer(name="fc1", output_dim=1200, neuron=Neurons.ReLU(),
                                weight_init = XavierInitializer,
                                #weight_cons = L2Cons(4.5),
                                bottoms=[:data], tops=[:fc1])
fc2_layer   = InnerProductLayer(name="fc2", output_dim=1200, neuron=Neurons.ReLU(),
                                weight_init = XavierInitializer,
                                weight_cons = L2Cons(4.5),
                                bottoms=[:fc1], tops=[:fc2])
fc3_layer   = InnerProductLayer(name="out", output_dim=10, bottoms=[:fc2],
                                weight_init = XavierInitializer,
                                weight_cons = L2Cons(4.5),
                                tops=[:out])
loss_layer  = SoftmaxLossLayer(name="loss", bottoms=[:out,:label])

# setup dropout for the different layers
# we use 20% dropout on the inputs and 50% dropout in the hidden layers
# as these values were previously found to be good defaults
drop_input  = DropoutLayer(name="drop_in", bottoms=[:data], ratio=0.2)
drop_fc1 = DropoutLayer(name="drop_fc1", bottoms=[:fc1], ratio=0.5)
drop_fc2  = DropoutLayer(name="drop_fc2", bottoms=[:fc2], ratio=0.5)

if parsed_args["useCUDA"]
  backend = GPUBackend()
else
  backend =CPUBackend()
end
init(backend)

common_layers = [fc1_layer, fc2_layer, fc3_layer]
drop_layers = [drop_input, drop_fc1, drop_fc2]
# put training net together, note that the correct ordering will automatically be established by the constructor
net = Net("NDSB_train", backend, [data_layer, common_layers..., drop_layers..., loss_layer])

base_path = "snapshots_dropout_fc"
# we let the learning rate decrease by 0.998 in each epoch (=600 batches of size 100)
# and let the momentum increase linearly from 0.5 to 0.9 over 500 epochs
# which is equivalent to an increase step of 0.0008
# training is done for 2000 epochs
params = SolverParameters(max_iter=600*2000, regu_coef=0.0,
                          mom_policy=MomPolicy.Linear(0.5, 0.0008, 600, 0.9),
                          lr_policy=LRPolicy.Step(0.1, 0.998, 600),
                          load_from=base_path)
solver = SGD(params)

setup_coffee_lounge(solver, save_into="$base_path/statistics.jld", every_n_iter=5000)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# save snapshots every 5000 iterations
add_coffee_break(solver, Snapshot(base_path), every_n_iter=5000)

# show performance on test data every 600 iterations (one epoch)
data_layer_test = HDF5DataLayer(name="test-data", source="data/test.txt", batch_size=100)
acc_layer = AccuracyLayer(name="test-accuracy", bottoms=[:out, :label], report_error=true)
test_net = Net("MNIST-test", backend, [data_layer_test, common_layers..., acc_layer])
add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=600)

solve(solver, net)

#Profile.init(int(1e8), 0.001)
#@profile solve(solver, net)
#open("profile.txt", "w") do out
#  Profile.print(out)
#end

destroy(net)
destroy(test_net)
shutdown(backend)
