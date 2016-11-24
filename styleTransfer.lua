require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'cutorch'
require 'cunn'
load_model = require 'load_model'

local function main()
  --load VGG 19 model;
  --this model has 16 cov layers and 4 avg pool layer
  --normalized weights
  model = load_model.loadVGG(128,128)

  -- load content image
  content_img = image.load("examples/golden_gate.jpg")
  content_img = preprocess(content_img):float()
  content_img = content_img:cuda()

  --create an white noise input image that has the same size as content image
  input_image = torch.DoubleTensor(content_img:size())
  input_image:uniform(0,1)
  input_image = preprocess(input_image):float()
  input_image = input_image:cuda()
  image.display(input_image)

  content_features_conv4_1 = model:forward(content_img):clone()
--[[  print(model:get(1).name)
  content_features_conv1_1 = model:get(2).output:clone()
  print(model:get(6).name)
  content_features_conv2_1 = model:get(7).output:clone()
  print(model:get(11).name)
  content_features_conv3_1 = model:get(12).output:clone()
  print(model:get(20).name)]]
  --content_features_conv4_1 = model:get(21).output:clone()
--[[  print(model:get(29).name)
  content_features_conv5_1 = model:get(30).output:clone()
]]
  criterion = nn.MSECriterion()
  criterion:cuda()
  local _, gradParams = model:getParameters()
  local function feval(x)
    gradParams:zero()

    -- Just run the network back and forth
    local yhat = model:forward(x)
    --local yhat = model:get(21).output:clone()
    local loss = criterion:forward(yhat,content_features_conv4_1)
    local gradLoss = criterion:backward(yhat,content_features_conv4_1)
    local gradInput = model:backward(x,gradLoss)

    collectgarbage()
    return loss, gradInput:view(gradInput:nElement())
  end

  optim_state = {
    learningRate = 0.03
    }
  for t = 1, 100 do
  	local x, losses = optim.adam(feval, input_image, optim_state)
  	print('Iteration number: '.. t ..'; Current loss: '.. losses[1])
  end

  output_image = deprocess(input_image:clone():double())
  image.display(output_image)
end

function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end
main()
--[[function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end]]









--[[

for i=1,2 do
  model:forward(input_image)
  output = model:get(20).output:clone()
  --local output = model:get(20).output


  local err = criterion:forward(output, targets)
  print(err)

  local gradLoss = criterion:backward(output, targets)
  local gradInput = model:backward(input_image, gradLoss)
  input_image = input_image - 0.1 * gradInput
end
image.display(input_image)
]]
