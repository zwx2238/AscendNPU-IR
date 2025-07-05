add_bishengir_conversion_library(BiShengIRLinalgToHFusion
  LinalgToHFusion.cpp

  ADDITIONAL_HEADER_DIRS
  ${BISHENGIR_MAIN_INCLUDE_DIR}/bishengir/Conversion/LinalgToHFusion

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_COMPONENTS
  Core
)

target_link_libraries(BiShengIRLinalgToHFusion PRIVATE ${BISHENG_IR_LIBS})
