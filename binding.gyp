{
  "targets": [
    {
      "target_name": "spatial_assembler",
      "sources": [
        "src/node_binding.cpp",
        "src/glb_core.cpp",
        "third_party/meshoptimizer/allocator.cpp",
        "third_party/meshoptimizer/vertexcodec.cpp",
        "third_party/meshoptimizer/indexcodec.cpp",
        "third_party/meshoptimizer/vertexfilter.cpp",
        "third_party/meshoptimizer/vfetchoptimizer.cpp"
      ],
      "include_dirs": [
        "src",
        "third_party/meshoptimizer",
        "third_party/cgltf"
      ],
      "defines": [
        "NAPI_VERSION=8",
        "NAPI_CPP_EXCEPTIONS"
      ],
      "cflags!": [
        "-fno-exceptions"
      ],
      "cflags_cc!": [
        "-fno-exceptions"
      ],
      "cflags": [
        "-O3"
      ],
      "cflags_cc": [
        "-O3",
        "-std=c++17"
      ],
      "xcode_settings": {
        "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
        "CLANG_CXX_LIBRARY": "libc++",
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "GCC_OPTIMIZATION_LEVEL": "3",
        "MACOSX_DEPLOYMENT_TARGET": "11.0"
      },
      "msvs_settings": {
        "VCCLCompilerTool": {
          "ExceptionHandling": 1,
          "Optimization": 2,
          "AdditionalOptions": [
            "/std:c++17",
            "/utf-8"
          ]
        }
      },
      "conditions": [
        [
          "OS==\"linux\"",
          {
            "cflags": [
              "-pthread"
            ],
            "ldflags": [
              "-pthread"
            ]
          }
        ]
      ]
    }
  ]
}
