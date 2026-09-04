{
  "targets": [
    {
      "target_name": "spatial_assembler",
      "sources": [
        "src/glb_exporter.cpp"
      ],
      "include_dirs": [],
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
        "-O3",
        "-ffast-math"
      ],
      "cflags_cc": [
        "-O3",
        "-std=c++17",
        "-ffast-math"
      ],
      "xcode_settings": {
        "CLANG_CXX_LANGUAGE_STANDARD": "c++17",
        "CLANG_CXX_LIBRARY": "libc++",
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "GCC_OPTIMIZATION_LEVEL": "3",
        "MACOSX_DEPLOYMENT_TARGET": "11.0",
        "OTHER_CPLUSPLUSFLAGS": [
          "-ffast-math"
        ]
      },
      "msvs_settings": {
        "VCCLCompilerTool": {
          "ExceptionHandling": 1,
          "Optimization": 2,
          "AdditionalOptions": [
            "/std:c++17",
            "/utf-8"
          ],
          "FloatingPointModel": 2
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
