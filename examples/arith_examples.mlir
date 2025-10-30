// 算术运算示例

// 示例1: 加法
func.func @test_add(%arg0: tensor<6x6xf32>, %arg1: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %ret = arith.addf %arg0, %arg1 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// 示例2: 乘法
func.func @test_mul(%arg0: tensor<6x6xf32>, %arg1: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %ret = arith.mulf %arg0, %arg1 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// 示例3: 减法
func.func @test_sub(%arg0: tensor<6x6xf32>, %arg1: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %ret = arith.subf %arg0, %arg1 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// 示例4: 除法
func.func @test_div(%arg0: tensor<6x6xf32>, %arg1: tensor<6x6xf32>) -> tensor<6x6xf32> {
  %ret = arith.divf %arg0, %arg1 : tensor<6x6xf32>
  return %ret : tensor<6x6xf32>
}

// 示例5: 组合运算 (a + b) * c
func.func @test_complex(%a: tensor<128xf32>, %b: tensor<128xf32>, %c: tensor<128xf32>) -> tensor<128xf32> {
  %0 = arith.addf %a, %b : tensor<128xf32>
  %1 = arith.mulf %0, %c : tensor<128xf32>
  return %1 : tensor<128xf32>
}
