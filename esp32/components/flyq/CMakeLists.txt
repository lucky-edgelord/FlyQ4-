idf_component_register(SRCS "flyq.c" INCLUDE_DIRS "include")
target_compile_definitions(${COMPONENT_LIB} PRIVATE FLYQ_IRAM=1)
target_compile_options(${COMPONENT_LIB} PRIVATE -O3)
