# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Declare a dialect in the include directory
function(add_bishengir_dialect dialect dialect_namespace)
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
  mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
  add_public_tablegen_target(BiShengIR${dialect}IncGen)
  add_dependencies(mlir-headers BiShengIR${dialect}IncGen)
endfunction()

# Declare the bishengir library associated with a dialect.
function(add_bishengir_library name)
  set_property(GLOBAL APPEND PROPERTY BISHENGIR_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_bishengir_library)

# Declare the bishengir library associated with a dialect.
function(add_bishengir_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY BISHENGIR_DIALECT_LIBS ${name})
  add_mlir_dialect_library(${ARGV})
endfunction(add_bishengir_dialect_library)

# Declare the bishengir library associated with a conversion.
function(add_bishengir_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY BISHENGIR_CONVERSION_LIBS ${name})
  add_mlir_conversion_library(${ARGV})
endfunction(add_bishengir_conversion_library)

# Declare the bishengir library associated with a translation.
function(add_bishengir_translation_library name)
  set_property(GLOBAL APPEND PROPERTY BISHENGIR_TRANSLATION_LIBS ${name})
  add_mlir_translation_library(${ARGV})
endfunction(add_bishengir_translation_library)

# Declare the bishengir library associated with an extension.
function(add_bishengir_extension_library name)
  set_property(GLOBAL APPEND PROPERTY BISHENGIR_EXTENSION_LIBS ${name})
  add_mlir_extension_library(${ARGV})
endfunction(add_bishengir_extension_library)

# Declare the bishengir target spec tablegen target.
function(bishengir_target_tablegen ofn)
  tablegen(BISHENGIR_TARGET_SPEC ${ARGV})
  set(TABLEGEN_OUTPUT
      ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)

  # Get the current set of include paths for this td file.
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
  list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
  # Filter out any empty include items.
  list(REMOVE_ITEM tblgen_includes "")

  # Build the absolute path for the current input file.
  if(IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE
        ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
  endif()

  # Append the includes used for this file to the tablegen_compile_commands
  # file.
  file(
    APPEND ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml
    "--- !FileInfo:\n" "  filepath: \"${LLVM_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
    "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${tblgen_includes}\"\n")
endfunction()

function(bishengir_options_tablegen ofn)
  tablegen(BISHENGIR_OPTIONS ${ARGV})
  set(TABLEGEN_OUTPUT
      ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)

  # Get the current set of include paths for this td file.
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
  list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
  # Filter out any empty include items.
  list(REMOVE_ITEM tblgen_includes "")

  # Build the absolute path for the current input file.
  if(IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE
        ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
  endif()

  # Append the includes used for this file to the tablegen_compile_commands
  # file.
  file(
    APPEND ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml
    "--- !FileInfo:\n" "  filepath: \"${LLVM_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
    "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${tblgen_includes}\"\n")
endfunction()

# Generate Documentation
function(add_bishengir_doc doc_filename output_file output_directory command)
  set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
  # The MLIR docs use Hugo, so we allow Hugo specific features here.
  tablegen(MLIR ${output_file}.md ${command} -allow-hugo-specific-features ${ARGN})
  set(GEN_DOC_FILE ${BISHENGIR_BINARY_DIR}/docs/${output_directory}${output_file}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  set_target_properties(${output_file}DocGen PROPERTIES FOLDER "BiShengIR/Tablegenning/Docs")
  add_dependencies(bishengir-doc ${output_file}DocGen)
endfunction()

# Adds an BiShengIR target for installation.
function(add_bishengir_publish_tool name)
  llvm_add_tool(MLIR ${ARGV})
  install(TARGETS ${name}
    COMPONENT bishengir-publish-products
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    # Note that CMake will create a directory like:
    #   objects-${CMAKE_BUILD_TYPE}/obj.LibName
    # and put object files there.
    OBJECTS DESTINATION lib${LLVM_LIBDIR_SUFFIX}
  )
endfunction()