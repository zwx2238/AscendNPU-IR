// 示例：使用 math.exp 计算指数函数
func.func @test_exp(%arg0: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %ret = math.exp %arg0 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// 示例：计算 sqrt
func.func @test_sqrt(%arg0: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %ret = math.sqrt %arg0 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// 示例：组合多个操作
func.func @compute(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = math.exp %arg0 : tensor<1024xf32>
  %1 = math.sqrt %0 : tensor<1024xf32>
  return %1 : tensor<1024xf32>
}
