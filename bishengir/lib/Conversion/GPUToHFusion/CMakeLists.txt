add_bishengir_conversion_library(BiShengIRGPUToHFusion
  GPUToHFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/GPUToHFusion

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core
)

target_link_libraries(BiShengIRGPUToHFusion PRIVATE ${BISHENG_IR_LIBS})