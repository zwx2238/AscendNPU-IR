func.func @compute(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = math.exp %arg0 : tensor<1024xf32>
  %1 = math.sqrt %0 : tensor<1024xf32>
  return %1 : tensor<1024xf32>
}
