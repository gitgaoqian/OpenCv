# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ros/opencv_files/c++/stereo_sgbm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ros/opencv_files/c++/stereo_sgbm

# Include any dependencies generated for this target.
include CMakeFiles/stereo_sgbm.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/stereo_sgbm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/stereo_sgbm.dir/flags.make

CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o: CMakeFiles/stereo_sgbm.dir/flags.make
CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o: stereo_sgbm.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ros/opencv_files/c++/stereo_sgbm/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o -c /home/ros/opencv_files/c++/stereo_sgbm/stereo_sgbm.cpp

CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ros/opencv_files/c++/stereo_sgbm/stereo_sgbm.cpp > CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.i

CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ros/opencv_files/c++/stereo_sgbm/stereo_sgbm.cpp -o CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.s

CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o.requires:
.PHONY : CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o.requires

CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o.provides: CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o.requires
	$(MAKE) -f CMakeFiles/stereo_sgbm.dir/build.make CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o.provides.build
.PHONY : CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o.provides

CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o.provides.build: CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o

# Object files for target stereo_sgbm
stereo_sgbm_OBJECTS = \
"CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o"

# External object files for target stereo_sgbm
stereo_sgbm_EXTERNAL_OBJECTS =

stereo_sgbm: CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o
stereo_sgbm: CMakeFiles/stereo_sgbm.dir/build.make
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
stereo_sgbm: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
stereo_sgbm: CMakeFiles/stereo_sgbm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable stereo_sgbm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stereo_sgbm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/stereo_sgbm.dir/build: stereo_sgbm
.PHONY : CMakeFiles/stereo_sgbm.dir/build

CMakeFiles/stereo_sgbm.dir/requires: CMakeFiles/stereo_sgbm.dir/stereo_sgbm.cpp.o.requires
.PHONY : CMakeFiles/stereo_sgbm.dir/requires

CMakeFiles/stereo_sgbm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/stereo_sgbm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/stereo_sgbm.dir/clean

CMakeFiles/stereo_sgbm.dir/depend:
	cd /home/ros/opencv_files/c++/stereo_sgbm && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ros/opencv_files/c++/stereo_sgbm /home/ros/opencv_files/c++/stereo_sgbm /home/ros/opencv_files/c++/stereo_sgbm /home/ros/opencv_files/c++/stereo_sgbm /home/ros/opencv_files/c++/stereo_sgbm/CMakeFiles/stereo_sgbm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/stereo_sgbm.dir/depend

