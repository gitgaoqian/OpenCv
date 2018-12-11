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
CMAKE_SOURCE_DIR = /home/ros/opencv_files/c++/stereo_match

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ros/opencv_files/c++/stereo_match

# Include any dependencies generated for this target.
include CMakeFiles/stereo_match.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/stereo_match.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/stereo_match.dir/flags.make

CMakeFiles/stereo_match.dir/stereo_match.cpp.o: CMakeFiles/stereo_match.dir/flags.make
CMakeFiles/stereo_match.dir/stereo_match.cpp.o: stereo_match.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ros/opencv_files/c++/stereo_match/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/stereo_match.dir/stereo_match.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/stereo_match.dir/stereo_match.cpp.o -c /home/ros/opencv_files/c++/stereo_match/stereo_match.cpp

CMakeFiles/stereo_match.dir/stereo_match.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stereo_match.dir/stereo_match.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ros/opencv_files/c++/stereo_match/stereo_match.cpp > CMakeFiles/stereo_match.dir/stereo_match.cpp.i

CMakeFiles/stereo_match.dir/stereo_match.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stereo_match.dir/stereo_match.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ros/opencv_files/c++/stereo_match/stereo_match.cpp -o CMakeFiles/stereo_match.dir/stereo_match.cpp.s

CMakeFiles/stereo_match.dir/stereo_match.cpp.o.requires:
.PHONY : CMakeFiles/stereo_match.dir/stereo_match.cpp.o.requires

CMakeFiles/stereo_match.dir/stereo_match.cpp.o.provides: CMakeFiles/stereo_match.dir/stereo_match.cpp.o.requires
	$(MAKE) -f CMakeFiles/stereo_match.dir/build.make CMakeFiles/stereo_match.dir/stereo_match.cpp.o.provides.build
.PHONY : CMakeFiles/stereo_match.dir/stereo_match.cpp.o.provides

CMakeFiles/stereo_match.dir/stereo_match.cpp.o.provides.build: CMakeFiles/stereo_match.dir/stereo_match.cpp.o

# Object files for target stereo_match
stereo_match_OBJECTS = \
"CMakeFiles/stereo_match.dir/stereo_match.cpp.o"

# External object files for target stereo_match
stereo_match_EXTERNAL_OBJECTS =

stereo_match: CMakeFiles/stereo_match.dir/stereo_match.cpp.o
stereo_match: CMakeFiles/stereo_match.dir/build.make
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
stereo_match: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
stereo_match: CMakeFiles/stereo_match.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable stereo_match"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stereo_match.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/stereo_match.dir/build: stereo_match
.PHONY : CMakeFiles/stereo_match.dir/build

CMakeFiles/stereo_match.dir/requires: CMakeFiles/stereo_match.dir/stereo_match.cpp.o.requires
.PHONY : CMakeFiles/stereo_match.dir/requires

CMakeFiles/stereo_match.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/stereo_match.dir/cmake_clean.cmake
.PHONY : CMakeFiles/stereo_match.dir/clean

CMakeFiles/stereo_match.dir/depend:
	cd /home/ros/opencv_files/c++/stereo_match && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ros/opencv_files/c++/stereo_match /home/ros/opencv_files/c++/stereo_match /home/ros/opencv_files/c++/stereo_match /home/ros/opencv_files/c++/stereo_match /home/ros/opencv_files/c++/stereo_match/CMakeFiles/stereo_match.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/stereo_match.dir/depend
