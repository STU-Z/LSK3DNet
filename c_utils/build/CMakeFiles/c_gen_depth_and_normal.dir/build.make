# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/zrb/Elements/Git_code/LSK3DNet/c_utils

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/zrb/Elements/Git_code/LSK3DNet/c_utils/build

# Include any dependencies generated for this target.
include CMakeFiles/c_gen_depth_and_normal.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/c_gen_depth_and_normal.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/c_gen_depth_and_normal.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/c_gen_depth_and_normal.dir/flags.make

CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.o: CMakeFiles/c_gen_depth_and_normal.dir/flags.make
CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.o: ../src/c_gen_depth_and_normal.cpp
CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.o: CMakeFiles/c_gen_depth_and_normal.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/zrb/Elements/Git_code/LSK3DNet/c_utils/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.o -MF CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.o.d -o CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.o -c /media/zrb/Elements/Git_code/LSK3DNet/c_utils/src/c_gen_depth_and_normal.cpp

CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/zrb/Elements/Git_code/LSK3DNet/c_utils/src/c_gen_depth_and_normal.cpp > CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.i

CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/zrb/Elements/Git_code/LSK3DNet/c_utils/src/c_gen_depth_and_normal.cpp -o CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.s

# Object files for target c_gen_depth_and_normal
c_gen_depth_and_normal_OBJECTS = \
"CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.o"

# External object files for target c_gen_depth_and_normal
c_gen_depth_and_normal_EXTERNAL_OBJECTS =

c_gen_depth_and_normal.cpython-39-x86_64-linux-gnu.so: CMakeFiles/c_gen_depth_and_normal.dir/src/c_gen_depth_and_normal.cpp.o
c_gen_depth_and_normal.cpython-39-x86_64-linux-gnu.so: CMakeFiles/c_gen_depth_and_normal.dir/build.make
c_gen_depth_and_normal.cpython-39-x86_64-linux-gnu.so: CMakeFiles/c_gen_depth_and_normal.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/zrb/Elements/Git_code/LSK3DNet/c_utils/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared module c_gen_depth_and_normal.cpython-39-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/c_gen_depth_and_normal.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /media/zrb/Elements/Git_code/LSK3DNet/c_utils/build/c_gen_depth_and_normal.cpython-39-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/c_gen_depth_and_normal.dir/build: c_gen_depth_and_normal.cpython-39-x86_64-linux-gnu.so
.PHONY : CMakeFiles/c_gen_depth_and_normal.dir/build

CMakeFiles/c_gen_depth_and_normal.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/c_gen_depth_and_normal.dir/cmake_clean.cmake
.PHONY : CMakeFiles/c_gen_depth_and_normal.dir/clean

CMakeFiles/c_gen_depth_and_normal.dir/depend:
	cd /media/zrb/Elements/Git_code/LSK3DNet/c_utils/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/zrb/Elements/Git_code/LSK3DNet/c_utils /media/zrb/Elements/Git_code/LSK3DNet/c_utils /media/zrb/Elements/Git_code/LSK3DNet/c_utils/build /media/zrb/Elements/Git_code/LSK3DNet/c_utils/build /media/zrb/Elements/Git_code/LSK3DNet/c_utils/build/CMakeFiles/c_gen_depth_and_normal.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/c_gen_depth_and_normal.dir/depend

