add_subdirectory(
  ${LLVM_MAIN_SRC_DIR}/../third-party/bishengir
  ${MLIR_BINARY_DIR}/../bishengir
  EXCLUDE_FROM_ALL)

include_directories(${LLVM_MAIN_SRC_DIR}/../third-party/bishengir)
include_directories(${MLIR_BINARY_DIR}/../bishengir)

set(BISHENG_IR_LIBS ${BISHENG_IR_INSTALL_PATH}/lib/libBiShengIR.so)

set(LLVM_LINK_COMPONENTS
  Support
  )


set(LIBS
  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRMlirOptMain
  MLIRPass
)

add_mlir_tool(bishengir-minimal-opt
  bishengir-minimal-opt.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  ${LIBS}
  )

target_link_libraries(bishengir-minimal-opt PRIVATE ${LIBS})
target_link_libraries(bishengir-minimal-opt PRIVATE ${BISHENG_IR_LIBS})

llvm_update_compile_flags(bishengir-minimal-opt)
mlir_check_all_link_libraries(bishengir-minimal-opt)
