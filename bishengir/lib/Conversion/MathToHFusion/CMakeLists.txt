add_bishengir_conversion_library(BiShengIRMathToHFusion
  MathToHFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/MathToHFusion

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core
)

target_link_libraries(BiShengIRMathToHFusion PRIVATE ${BISHENG_IR_LIBS})
