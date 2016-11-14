require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'cutorch'
require 'cunn'
load_model = require 'load_model'


model = load_model.loadVGG(128,128)


params, gradParams = model:getParameters()

local optimState = {learningRate = 0.01}

function feval(params)
   gradParams:zero()

   local outputs = model:forward(batchInputs)
   local loss = criterion:forward(outputs, batchLabels)
   local dloss_doutputs = criterion:backward(outputs, batchLabels)
   model:backward(batchInputs, dloss_doutputs)

   return loss, gradParams
end
optim.sgd(feval, params, optimState)
