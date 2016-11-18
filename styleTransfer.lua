require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'cutorch'
require 'cunn'
load_model = require 'load_model'

--load VGG 19 model;
--this model has 16 cov layers and 4 avg pool layer
--normalized weights
model = load_model.loadVGG(128,128)
-- load content image
content_img = image.load("examples/golden_gate.jpg")
content_img = content_img:cuda()
--create an white noise input image that has the same size as content image
input_image = torch.CudaTensor(content_img:size())
input_image:uniform(0,1)
image.display(input_image)

model:forward(content_img)
print(model:get(1).name)
content_features_conv1_1 = model:get(1).output
print(model:get(6).name)
content_features_conv2_1 = model:get(6).output
print(model:get(11).name)
content_features_conv3_1 = model:get(11).output
print(model:get(20).name)
content_features_conv4_1 = model:get(20).output
print(model:get(29).name)
content_features_conv5_1 = model:get(29).output

criterion = nn.MSECriterion()
criterion:cuda()


--targets = model:get(20).output
targets = model:forward(content_img)
for i=1,10 do


  local output = model:forward(input_image)
--local output = model:get(20).output
--[[  input_features_conv1_1 = model:get(1).output
  input_features_conv2_1 = model:get(6).output
  input_features_conv3_1 = model:get(11).output
  input_features_conv4_1 = model:get(20).output
  input_features_conv5_1 = model:get(29).output]]
  local err = criterion:forward(output, targets)
  local gradLoss = criterion:backward(output, targets)
  local gradInput = model:backward(input, gradLoss)

  input_image = input_image - 0.1 * gradInput

end
