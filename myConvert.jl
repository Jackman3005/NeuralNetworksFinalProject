# convert binary into HDF5 data

using HDF5, JLD
using Images

#Specify the directory you want to search
cd("C:/Users/Jack/Git/CSCI 7222 - Neural Networks/data/train")

mainDir = readdir()

labels = Array(Float32,(151684))
labelNames = Array(String,(0))
images = Array(Float32,(100,100,1,151684))
labelCnt = 0
numImagesProcessed = 1
for labelDir in mainDir
  if !contains(labelDir,".") #Make sure not a file
    #Map names to number

    cat(1,labelNames,labelDir)
    println("Processing:",labelDir)
    subDir = readdir(labelDir)
    for imgFile in subDir
      if !contains(imgFile, ".db")
        labels[numImagesProcessed] = labelCnt
        img = imread(labelDir * "/" * imgFile)
        images[:,:,:,numImagesProcessed] =data(img)
        if (numImagesProcessed % 1000 == 0)
          println("Processing Image # $numImagesProcessed")
        end
        numImagesProcessed += 1
      end
    end

    labelCnt += 1
  end
end
randomIndices = randperm(numImagesProcessed)
smallSubset = randomIndices[1:1000]
medSubset = randomIndices[1:10000]

save("../trainData", "label", labels, "data", images, "labelNames", labelNames)
save("../trainDataMedSubset","label",labels[medSubset],"data",images[:,:,:,medSubset],"labelNames",labelNames)
save("../trainDataSmallSubset","label",labels[smallSubset],"data",images[:,:,:,smallSubset],"labelNames",labelNames)
