cmake_minimum_required(VERSION 3.18)
project(xara_mesh_triangle C)

add_library(triangle_lib SHARED
    src/c/triangle.c
)

target_compile_definitions(triangle_lib PRIVATE
    TRILIBRARY
    REAL=double
    VOID=void
    NO_TIMER
)

# Windows: export symbols
if (MSVC)
  set_target_properties(triangle_lib PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# install(TARGETS triangle_lib
#         COMPONENT triangle_lib
#         LIBRARY DESTINATION xara_mesh/schewchuk
#         RUNTIME DESTINATION xara_mesh/schewchuk      # .dll
#         ARCHIVE DESTINATION xara_mesh/schewchuk)     # static lib (ignored on wheel)

install(TARGETS triangle_lib
        LIBRARY DESTINATION xara_mesh/schewchuk
        RUNTIME DESTINATION xara_mesh/schewchuk
        ARCHIVE DESTINATION xara_mesh/schewchuk)
