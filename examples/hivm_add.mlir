// HIVM 层向量加法示例
// 这是一个完整的 HIVM 层代码，可以直接编译到硬件

module {
  // DEVICE 函数：在 NPU 上执行
  func.func @add(
    %arg0: memref<16xf32, #hivm.address_space<gm>>,  // 全局内存输入1
    %arg1: memref<16xf32, #hivm.address_space<gm>>,  // 全局内存输入2
    %arg2: memref<16xf32, #hivm.address_space<gm>>   // 全局内存输出
  ) attributes {
    hacc.entry,                                       // 标记为入口函数
    hacc.function_kind = #hacc.function_kind<DEVICE>  // 设备函数
  } {
    // 1. 在 UB (Unified Buffer) 中分配缓冲区
    %buf0 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>

    // 2. 从全局内存加载到 UB
    hivm.hir.load ins(%arg0 : memref<16xf32, #hivm.address_space<gm>>)
                  outs(%buf0 : memref<16xf32, #hivm.address_space<ub>>)

    // 3. 加载第二个输入
    %buf1 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%arg1 : memref<16xf32, #hivm.address_space<gm>>)
                  outs(%buf1 : memref<16xf32, #hivm.address_space<ub>>)

    // 4. 向量加法（在 UB 中执行）
    %buf_result = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    hivm.hir.vadd ins(%buf0, %buf1 : memref<16xf32, #hivm.address_space<ub>>,
                                      memref<16xf32, #hivm.address_space<ub>>)
                  outs(%buf_result : memref<16xf32, #hivm.address_space<ub>>)

    // 5. 存储回全局内存
    hivm.hir.store ins(%buf_result : memref<16xf32, #hivm.address_space<ub>>)
                   outs(%arg2 : memref<16xf32, #hivm.address_space<gm>>)

    return
  }
}
