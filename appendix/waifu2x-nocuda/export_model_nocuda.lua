require 'nn'
require './lib/LeakyReLU'
local cjson = require "cjson"

local model = torch.load(arg[1], "ascii")

local jmodules = {}

for i=1, table.maxn(model.modules), 1 do
    local module = model.modules[i]
    if tostring(module) == "nn.SpatialConvolution" then
        local jmod = {
            kW = module.kW,
            kH = module.kH,
            nInputPlane = module.nInputPlane,
            nOutputPlane = module.nOutputPlane,
            bias = torch.totable(module.bias:float()),
            weight = torch.totable(module.weight:float())
        }
        table.insert(jmodules, jmod)
    end
end

io.write(cjson.encode(jmodules))
