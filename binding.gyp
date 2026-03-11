{
  "targets": [
    {
      "target_name": "spatial_assembler",
      "sources": ["src/glb_exporter.cpp"],
      "include_dirs": [],
      "cflags!": ["-fno-exceptions"],
      "cflags_cc!": ["-fno-exceptions"],
      "cflags": ["-O3", "-ffast-math", "-pthread"],
      "cflags_cc": ["-O3", "-ffast-math", "-std=c++17", "-pthread"],
      "ldflags": ["-pthread"],
      "defines": ["NAPI_CPP_EXCEPTIONS"]
    }
  ]
}
