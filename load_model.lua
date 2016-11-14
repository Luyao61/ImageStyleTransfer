require 'torch'
require 'nn'
require 'loadcaffe'

local module = {}

function module.loadVGG(height, width)
  model = loadcaffe.load('models/VGG_ILSVRC_19_layers_deploy.prototxt', 'models/VGG_ILSVRC_19_layers.caffemodel', 'nn')

  local model_size = #model
  for i = 1, model_size do
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
    elseif (layer_type == 'nn.SpatialConvolution') then
      -- normalize the network
      --[["scale the weights such that the mean activation of each convolutional
          filter over images and positions equal to one"
          paper "Image Style Transfer using CNN", page 2416]]
      local filter_size = layer.weight:size(2)*layer.weight:size(3)*layer.weight:size(4)
      for j = 1,layer.weight:size(1) do
        local mean = layer.weight[1]:sum()/filter_size
        layer.weight[1]:div(mean)
      end
      print(("Normalize layer: %s"):format(name))
    end
    --print(('%s; %s'):format(name,layer_type))
  end
  model:remove(46)
  model:remove(45)
  model:remove(44)
  model:remove(43)
  model:remove(42)
  model:remove(41)
  model:remove(40)
  model:remove(39)
  model:remove(38)

  model:cuda()
  return model
end

return module
