// 示例：使用 arith.addf 进行浮点数加法
func.func @test_add(%arg0: tensor<6x6xi64>, %arg1: tensor<6x6xi64>) -> tensor<6x6xi64> {
  %ret = arith.addi %arg0, %arg1 : tensor<6x6xi64>
  return %ret : tensor<6x6xi64>
}