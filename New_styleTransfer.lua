require 'torch'
require 'nngraph'
require 'loadcaffe'
require 'cutorch'
require 'cunn'
require 'image'
require 'optim'
require 'nn'
--torch.setdefaulttensortype('torch.FloatTensor')

function main()

  --first part; load pretrained VGG_ILSVRC_19_layers
  --build model for our code;
  vgg19 = loadcaffe.load('models/VGG_ILSVRC_19_layers_deploy.prototxt', 'models/VGG_ILSVRC_19_layers.caffemodel'):float()
  local content_layers = {'relu4_2'}
  local style_layers = {'relu1_1','relu2_1','relu3_1','relu4_1','relu5_1'}
  content_layer_index = 1
  style_layer_index = 1
  content_weight = 0.001
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
  --      output_layers[#output_layers + 1] = nn.MulConstant(content_weight/(128*256*256/2^32))(model_layers[#model_layers])
        output_layers[#output_layers + 1] = nn.MulConstant(content_weight/(128*256*256/2^32))(model_layers[#model_layers])
        content_layer_index = content_layer_index + 1
      elseif name == style_layers[style_layer_index] then
        output_layers[#output_layers + 1] = nn.MulConstant(style_weight/(128*256*256/2^style_layer_index))(GramMatrix()(model_layers[#model_layers]))
--        output_layers[#output_layers + 1] = nn.MulConstant(style_weight/(128*256*256/2^style_layer_index))(GramMatrix()(model_layers[#model_layers]))
        style_layer_index = style_layer_index + 1
      end
    end
  end

  local model = nn.gModule({model_layers[1]}, output_layers)
  model:cuda()

  -- load content image
  content_img = image.load("examples/gg.png")
  --content_img = image.crop(content_img,"c",300,300)
  content_img = preprocess(content_img):float()
  content_img = content_img:cuda()

  -- load style image
  style_img = image.load("examples/stary.png")
  --style_img = image.crop(style_img,"c",300,300)
  style_img = preprocess(style_img):float()
  style_img = style_img:cuda()

  --create an white noise input image that has the same size as content image
  input_image = torch.DoubleTensor(content_img:size())
  input_image:uniform(0,1)
  input_image = preprocess(input_image):float()
  input_image = input_image:cuda()

  -- creat target; content features and style features which are used to
  -- create new style image.
  local target = {}

  local content_output
  content_output = model:forward(content_img)
  target[5] = content_output[5]:clone()

  local style_output
  style_output = model:forward(style_img)
  target[1] = style_output[1]:clone()
  target[2] = style_output[2]:clone()
  target[3] = style_output[3]:clone()
  target[4] = style_output[4]:clone()
  target[6] = style_output[6]:clone()


  criterion = nn.ParallelCriterion()
  for i=1,6 do
    criterion:add(nn.MSECriterion())
  end
  criterion:cuda()

  local _, gradParams = model:getParameters()
  local function feval(x)
    gradParams:zero()

    local yhat = model:forward(x)
    local loss = criterion:forward(yhat,target)
    local gradLoss = criterion:backward(yhat,target)
    local gradInput = model:backward(x,gradLoss)

    collectgarbage()
    return loss, gradInput:view(gradInput:nElement())
  end

  optim_state = {
    learningRate = 1
  }

  for t = 1, 20 do
    local x, losses = optim.lbfgs(feval, input_image, optim_state)
    print('Iteration number: '.. t ..'; Current loss: '.. losses[1])
  end

  output_image = deprocess(input_image:clone():double())
  image.display(output_image)
  image.save("out.jpg",output_image)

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
main()
