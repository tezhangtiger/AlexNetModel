library(MicrosoftML)
library(readr)

#net definition
alexNetDef = ""
for (i in 1:10) {
  zipf=paste0("alexNetDef", as.character(i), ".zip")
  unzip(zipf)
  txtf=paste0("alexNetDef", as.character(i), ".txt")
  alexNetDef=paste0(alexNetDef, read_file(txtf))
}

imgSize = 3*227*227

#one data sample
inData = cbind(label=factor(1, levels = 1:1000), data.frame(matrix(rep(imgSize),1,imgSize)))

#import to xdf format
inDataXdf="inData.xdf"
rxImport(inData = inData, outFile = inDataXdf, stringsAsFactors=TRUE, overwrite = TRUE)
inData=RxXdfData(inDataXdf)

#define formula
form = label~.

#define optimizer
sgdOptimizer = sgd(learningRate = 0.01, momentum = 0.9, weightDecay = 0.00005,
                   nag = TRUE, lRateRedRatio = 0.1, lRateRedErrorRatio = 0)

#create model
model_alexnet = rxNeuralNet(formula = form, data = inData, numIterations = 0,
                            type = "multi", netDefinition = alexNetDef,
                            initWtsDiameter = 0.1,
                            miniBatchSize = 128, acceleration = "gpu",
                            optimizer = sgdOptimizer)
