using HDF5
using JLD

cd("/home/brett/GitProjects/NeuralNetworksFinalProject/")

fobj = open("testData.txt","w")

data = load("snapshots/Outputs.jld")

writecsv("testOutput.csv",data["ip1"][:,1:20500]')
