using HDF5
using JLD

cd("/home/brett/GitProjects/NeuralNetworksFinalProject/")

data = load("snapshots/Outputs.jld")

writecsv("testOutput.csv",data["ip1"][:,1:130401]')
