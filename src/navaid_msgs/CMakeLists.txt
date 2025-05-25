cmake_minimum_required(VERSION 3.8)
project(navaid_msgs)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED) # For generating message code

# Generate C/C++ and Python code from .msg files
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LedArrayState.msg"
  # If your message depended on other interfaces (e.g. std_msgs/Header), list them as:
  # Example: DEPENDENCIES std_msgs
)

# Install the .msg files themselves - good practice
install(
    DIRECTORY msg
    DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# ament_package() handles exporting interfaces for packages that are members of
# rosidl_interface_packages group (declared in package.xml).
# Explicit ament_export_targets for __py is sometimes needed for very specific
# scenarios or older ament versions, but let's try relying on the standard mechanism first.
ament_package(
    # If you had C++ libraries to export: LIBRARIES ${my_cpp_lib_target}
    # If you had CMake config files to export for C++ downstream: CONFIG_EXTRAS my_extras.cmake
)