# BiShengIR Project.

set(BISHENG_IR_INSTALL_PATH "" CACHE STRING
      "install path for pre-built required for bishengir")

if(${BISHENG_IR_INSTALL_PATH} STREQUAL "")
  message(FATAL_ERROR "Must provide the path to pre-built.")
endif()

set(BISHENG_IR_YAML_GEN_EXE ${BISHENG_IR_INSTALL_PATH}/bin/bishengir-yaml-gen)

set(MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LLVM_SOURCE_DIR ${LLVM_MAIN_SRC_DIR})
set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
set(MLIR_INCLUDE_DIRS ${MLIR_MAIN_SRC_DIR}/include)

set(MLIR_CMAKE_DIR ${MLIR_MAIN_SRC_DIR}/cmake/modules)
set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
set(BISHENGIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/bishengir)
include_directories(${MLIR_TABLEGEN_OUTPUT_DIR})
include_directories(${BISHENGIR_TABLEGEN_OUTPUT_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddMLIRPython)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})

link_directories(${LLVM_BUILD_LIBRARY_DIR})

add_definitions(${LLVM_DEFINITIONS})

set(BISHENGIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BISHENGIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(BISHENGIR_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")

set(MLIR_ENABLE_BINDINGS_PYTHON 0 CACHE BOOL
      "Enables building of Python bindings.")

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

add_subdirectory(bishengir)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY bishengir
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT bishengir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "LICENSE.TXT"
    )

  install(DIRECTORY ${BISHENGIR_BINARY_DIR}/bishengir
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT bishengir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.gen"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "config.h" EXCLUDE
    )

  if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-bishengir-headers
                             DEPENDS bishengir-headers
                             COMPONENT bishengir-headers)
  endif()
endif()
