add_library(lbp STATIC 
            bipgraph.cpp
            bp.cpp
            factor.cpp 
            factorgraph.cpp 
            graph.cpp
            index.cpp 
            prob.cpp
            properties.cpp
            smallset.cpp
            var.cpp
            varset.cpp)

# # Set CUDA properties (compute architecture, etc.)
# set_target_properties(lbp PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES 75  # Set appropriate architectures for your GPUs
# )

target_link_libraries(lbp PUBLIC kernel)
target_include_directories(lbp PUBLIC ${PROJECT_SOURCE_DIR}/include)
