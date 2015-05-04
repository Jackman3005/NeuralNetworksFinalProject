using PyPlot
using HDF5
using JLD

cd("/home/brett/GitProjects/NeuralNetworksFinalProject/")

data = load("statistics.jld")

srtIter = sort(collect(keys(data["statistics"]["accuracy-accuracy"])))

acc = []
err = []
epochs = []
lr = []
mom = []

for i in srtIter
    acc = cat(1,acc,data["statistics"]["accuracy-accuracy"][i])
    err = cat(1,err,data["statistics"]["accuracy-error"][i])
    mom = cat(1,mom,data["statistics"]["momentum"][i])
    lr = cat(1,lr,data["statistics"]["learning-rate"][i])
    epochs = cat(1,epochs,int(i))
end

#convert from batches to epocs
epochs = epochs.*100 ./ 150000

#Each has two identical entries at the beginning, this removes the first
#shift!(acc)
#shift!(err)
#shift!(mom)
#shift!(lr)

plot(epochs,acc)
title("Accuracy vs. Epoch")
xlabel("Epochs")
ylabel("Percent")

plot(epochs,err)
title("Error vs. Epoch")
xlabel("Epochs")
ylabel("Percent")

plot(epochs,mom)
title("Momentum vs. Epoch")
xlabel("Epochs")
ylabel("Momentum")

plot(epochs,lr)
title("Learning Rate vs. Epoch")
xlabel("Epochs")
ylabel("Learning Rate")