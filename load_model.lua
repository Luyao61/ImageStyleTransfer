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
      local kW, kH = layer.kW, layer.kH
      local dW, dH = layer.dW, layer.dH
      local padW, padH = layer.padW, layer.padH

      local avg_pooling = nn.SpatialAveragePooling(kW, kH, dW, dH, padW, padH)
      model:remove(i)
      model:insert(avg_pooling, i)
      print(("Replace Max Pool layer #%d with avg pool layer"):format(i))
    elseif (layer_type == 'nn.SpatialConvolution') then
      local a =0
    end
    --print(('%s; %s'):format(name,layer_type))
  end
  model:remove(46)
  model:remove(45)
  model:remove(45)
  model:remove(43)
  model:remove(42)
  model:remove(41)
  model:remove(40)
  model:remove(39)
  model:remove(38)


  model:cuda()
  print(model)
  return model
end

return module
