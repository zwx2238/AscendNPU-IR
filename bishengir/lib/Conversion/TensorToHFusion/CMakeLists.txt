add_bishengir_conversion_library(BiShengIRTensorToHFusion
  TensorToHFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/TensorToHFusion

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core
)

target_link_libraries(BiShengIRTensorToHFusion PRIVATE ${BISHENG_IR_LIBS})
