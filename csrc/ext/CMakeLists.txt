add_library(ext STATIC
            causal_bp.cu
            causal_factor.cpp
            causal_fg.cpp)

target_link_libraries(ext PUBLIC lbp)
target_include_directories(ext PUBLIC ${PROJECT_SOURCE_DIR}/include)
