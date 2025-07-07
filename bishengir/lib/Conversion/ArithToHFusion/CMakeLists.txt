add_bishengir_conversion_library(BiShengIRArithToHFusion
  ArithToHFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/ArithToHFusion

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core
)

target_link_libraries(BiShengIRArithToHFusion PRIVATE ${BISHENG_IR_LIBS})