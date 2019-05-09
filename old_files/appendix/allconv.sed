:loop
N
$!b loop
s/16\ntorch.CudaTensor/17\ntorch.FloatTensor/g
s/17\ntorch.CudaStorage/18\ntorch.FloatStorage/g
s/24\ncudnn.SpatialConvolution/21\nnn.SpatialConvolution/g
