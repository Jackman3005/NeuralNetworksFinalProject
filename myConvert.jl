# convert binary into HDF5 data

using HDF5, JDL
using Images

#Specify the directory you want to search
cd("/home/brett/.julia/v0.3/Mocha/examples/mnist")

mainDir = readdir()

labels = Array(Int,(0))
labelNames = Array(String,(0))
images = Array(Float64,(100,100,1,0))
labelCnt = 0

for labelDir in mainDir
  if !contains(labelDir,".") #Make sure not a file
    #Map names to number

    cat(1,labelNames,labelDir)

    subDir = readdir(labelDir)
    for imgFile in subDir
      cat(1,labels,labelCnt)
      img = imread(imgFile)
      images = cat(4,images,data(img))
    end

    labelCnt += 1
end

save("trainData", "label", labels, "data", images, "labelNames", labelNames)
