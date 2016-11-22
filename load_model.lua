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
      if (name == style_layers[style_layer_index]) then
        local gram = GramMatrix():float()


        style_layer_index = style_layer_index + 1
      end
    end
  end

  return model
end

return module
