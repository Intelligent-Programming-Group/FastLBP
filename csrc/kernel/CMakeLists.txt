add_library(kernel STATIC
            causal_bp_parall.cu
            causal_bp_seq.cu)

# # Set CUDA properties (compute architecture, etc.)
# set_target_properties(kernel PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_ARCHITECTURES 75  # Set appropriate architectures for your GPUs
# )

# Link CUDA library to ensure correct linking
target_include_directories(kernel PUBLIC ${PROJECT_SOURCE_DIR}/include)
