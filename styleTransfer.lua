require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'cutorch'
require 'cunn'

load_model = require 'load_model'

load_model.loadVGG(128,128)
