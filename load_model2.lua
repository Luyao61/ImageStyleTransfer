require 'torch'
require 'nn'
require 'loadcaffe'

local module = {}

function module.loadVGG(height, width)
  vgg19 = loadcaffe.load('models/VGG_ILSVRC_19_layers_deploy.prototxt', 'models/VGG_ILSVRC_19_layers.caffemodel', 'nn')

  local model = nn.sequential()
  local content_layers = ['relu4_2']
  local style_layers = ['relu1_1','relu2_1','relu3_1','relu4_1','relu5_1']
  content_layer_index = 1
  style_layer_index = 1
  content_weight = 0.5
  style_weight = 0.5

  local model_size = #model
  for i = 1, model_size do
    if content_layer_index <= #content_layers or style_layer_index <= #style_layers then
      local layer = model:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)

      if(layer_type == 'nn.SpatialMaxPooling') then
        -- replace maxpooling layer to averagepooling
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local padW, padH = layer.padW, layer.padH

        local avg_pooling = nn.SpatialAveragePooling(kW, kH, dW, dH, padW, padH)
        model:remove(i)
        model:insert(avg_pooling, i)
        print(("Replace Max Pool layer #%d with avg pool layer"):format(i))
      else
        model:add(layer)
      end
      if (name == content_layers[content_layer_index]) then
        local target = net:forward(content_image):clone()
        local loss_module = nn.ContentLoss(content_weight,target.false)
        model:add(loss_module)
        table.insert(content_losses, loss_module):float()
        content_layer_index = content_layer_index + 1
      end
--[[      if (name == style_layers[style_layer_index]) then
        local gram = GramMatrix():float()
        local target = nil
        local target_features = net:forward(style_image):clone()

        style_layer_index = style_layer_index + 1
      end]]
    end
  end

  return model
end


--create a content loss module to store content loss
--http://torch.ch/docs/developer-docs.html
-- weight is the content loss weight
-- target is the ocntent features, ie. ground truth
local ContentLoss, Parent = torch.class('nn.ContentLoss', 'nn.Module')
function ContentLoss:__init(weight, target)
   Parent.__init(self)
   self.weight = weight
   self.target = target
   self.loss = 0
   self.criterion = nn.MSECriterion()
   end

function ContentLoss:updateOutput(input)
  self.loss = self.criterion(input, target) * self.weight
  return input
end

function ContentLoss:updateGradInput(input, gradOutput)
  self.gradInput = self.crit:backward(input, self.target)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

local StyleLoss, Parent = torch.class('nn.StyleLoss', 'nn.Module')
function StyleLoss:__init(weight, target)
   Parent.__init(self)
   self.weight = weight
   self.target = target
   self.loss = 0
   self.criterion = nn.MSECriterion()

   self.G = 0
   self.GramMatrix = GramMatrix()
   end

function StyleLoss:updateOutput(input)
  self.G = self.GramMatrix:forward(input)
  self.G:div(input:nElement())
  self.loss = self.criterion(input, target) * self.weight
  return input
end

function StyleLoss:updateGradInput(input, gradOutput)
  self.gradInput = self.crit:backward(input, self.target)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end
return module
