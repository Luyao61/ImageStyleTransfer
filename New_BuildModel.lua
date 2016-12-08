require 'torch'
require 'nngraph'
require 'loadcaffe'
require 'cutorch'
require 'cunn'
--first part; load pretrained VGG_ILSVRC_19_layers
--build model for our code;
vgg19 = loadcaffe.load('models/VGG_ILSVRC_19_layers_deploy.prototxt', 'models/VGG_ILSVRC_19_layers.caffemodel'):float()
local content_layers = {'relu4_2'}
local style_layers = {'relu1_1','relu2_1','relu3_1','relu4_1','relu5_1'}
content_layer_index = 1
style_layer_index = 1
content_weight = 0.5
style_weight = 0.5

model_layers = {}
output_layers = {}
model_layers[1] = nn.Identity()()
for i=1, #vgg19 do
  if content_layer_index <= #content_layers or style_layer_index <= #style_layers then
    local layer = vgg19:get(i)
    local name = layer.name
    local layer_type = torch.type(layer)


    if(layer_type == 'nn.SpatialMaxPooling') then
      local kW, kH = layer.kW, layer.kH
      local dW, dH = layer.dW, layer.dH
      local padW, padH = layer.padW, layer.padH
      local avg_pooling = nn.SpatialAveragePooling(kW, kH, dW, dH, padW, padH)
      model_layers[#model_layers+1] = avg_pooling(model_layers[#model_layers])
    else
      model_layers[#model_layers+1] = layer(model_layers[#model_layers])
    end

    if name == content_layers[content_layer_index] then
      output_layers[#output_layers + 1] = nn.Identity()(model_layers[#model_layers])
      content_layer_index = content_layer_index + 1
    elseif name == style_layers[style_layer_index] then
      output_layers[#output_layers + 1] = nn.Identity()(model_layers[#model_layers])
      style_layer_index = style_layer_index + 1
    end
  end
end

local model = nn.gModule({model_layers[1]}, output_layers)
model:cuda()

torch.save("style_transfer_model.t7",model)
