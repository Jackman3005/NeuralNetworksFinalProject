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

        "--trainFile", "-F"
            help = "path to the training data file you want to use"
            arg_type = String
            default = ""

        "--testFile", "-t"
            help = "path to the test data file you want to use"
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

base_path = parsed_args["basePath"]
train_path = parsed_args["trainFile"]
test_path = parsed_args["testFile"]

using Mocha

#####################################################
#####             Begin Netowrk                ######
#####################################################

# fix the random seed to make results reproducable
#srand(12345678)

data_layer  = AsyncHDF5DataLayer(name="train-data", source=train_path, batch_size = 100)

#Layer for reduction of image size
IR = (50,50,1)
fc1_layer   = InnerProductLayer(name="fc1", output_dim=prod(IR[1:2]), neuron=Neurons.ReLU(),
    weight_init = XavierInitializer(),bottoms=[:data], tops=[:fc1])

#Convolution layer needs 4D tensor so we need to reshape outputs from InnerProductLayer (the fourth dimension is implicit)
reshape_layer = ReshapeLayer(shape=IR,bottoms=[:fc1], tops=[:rs1])

conv1_layer = ConvolutionLayer(name="conv1", n_filter=32, kernel=(5,5), pad=(2,2),
    stride=(1,1), filter_init=XavierInitializer(), bottoms=[:rs1], tops=[:conv1])
pool1_layer = PoolingLayer(name="pool1", kernel=(3,3), stride=(2,2), neuron=Neurons.ReLU(),
    bottoms=[:conv1], tops=[:pool1])
norm1_layer = LRNLayer(name="norm1", kernel=3, scale=5e-5, power=0.75, mode=LRNMode.WithinChannel(),
    bottoms=[:pool1], tops=[:norm1])

conv2_layer = ConvolutionLayer(name="conv2", n_filter=32, kernel=(5,5), pad=(2,2),
    stride=(1,1), filter_init=XavierInitializer(), bottoms=[:norm1], tops=[:conv2], neuron=Neurons.ReLU())
pool2_layer = PoolingLayer(name="pool2", kernel=(3,3), stride=(2,2), pooling=Pooling.Mean(),
    bottoms=[:conv2], tops=[:pool2])
norm2_layer = LRNLayer(name="norm2", kernel=3, scale=5e-5, power=0.75, mode=LRNMode.WithinChannel(),
    bottoms=[:pool2], tops=[:norm2])

conv3_layer = ConvolutionLayer(name="conv3", n_filter=64, kernel=(5,5), pad=(2,2),
    stride=(1,1), filter_init=XavierInitializer(), bottoms=[:norm2], tops=[:conv3], neuron=Neurons.ReLU())
pool3_layer = PoolingLayer(name="pool3", kernel=(3,3), stride=(2,2), pooling=Pooling.Mean(),
    bottoms=[:conv3], tops=[:pool3])

ip1_layer   = InnerProductLayer(name="ip1", output_dim=121, weight_init=XavierInitializer(),
    bottoms=[:pool2], tops=[:ip1])

HDF5Output = HDF5OutputLayer(filename="$base_path/snapshots/Outputs.jld",bottoms=[:ip1], force_overwrite=true)
loss_layer  = SoftmaxLossLayer(name="softmax", bottoms=[:ip1, :label])
acc_layer   = AccuracyLayer(name="accuracy", bottoms=[:ip1, :label],report_error=true)

common_layers = [fc1_layer, reshape_layer, conv1_layer, pool1_layer, ip1_layer, norm1_layer, conv2_layer, pool2_layer]


# setup dropout for the different layers
# we use 10% dropout on the inputs and 50% dropout in the hidden layers
# as these values were previously found to be good defaults
drop_input  = DropoutLayer(name="drop_in", bottoms=[:data], ratio=0.1)
drop_norm1  = DropoutLayer(name="drop_norm1", bottoms=[:norm1], ratio=0.5)
#drop_norm2  = DropoutLayer(name="drop_norm2", bottoms=[:norm2], ratio=0.5)
drop_ip1 = DropoutLayer(name="drop_ip1", bottoms=[:ip1], ratio=0.5)


if parsed_args["useCUDA"]
  backend = GPUBackend()
else
  backend =CPUBackend()
end
init(backend)

drop_layers = [drop_norm1, drop_ip1]
#drop_layers = []
# put training net together, note that the correct ordering will automatically be established by the constructor
net = Net("NDSB_train", backend, [data_layer, common_layers..., drop_layers..., loss_layer])

#println(net)

num_iters = 1000000

params = SolverParameters(max_iter=num_iters, regu_coef=0.0,
                          mom_policy=MomPolicy.Fixed(.9),
                          #mom_policy=MomPolicy.Linear(0.5, 0.0008, num_iters, 0.9),
                          lr_policy=LRPolicy.Step(.001,0.98,num_iters),
                          load_from="$base_path/data/snapshot-985000.jld")


solver = Nesterov(params)

data_layer_test = HDF5DataLayer(name="test-data", source=test_path, batch_size=100)
test_net = Net("NDSB-test", backend, [data_layer_test, drop_layers..., common_layers..., HDF5Output, acc_layer])

add_coffee_break(solver, ValidationPerformance(test_net), every_n_iter=1)

solve(solver, net)

destroy(net)
destroy(test_net)
shutdown(backend)
