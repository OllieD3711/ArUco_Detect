# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_SOURCE_DIR = /home/jake/Documents/ArUco_Detect

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jake/Documents/ArUco_Detect/pbuild

# Utility rule file for jvpkg.

# Include the progress variables for this target.
include CMakeFiles/jvpkg.dir/progress.make

CMakeFiles/jvpkg:
	cd /home/jake/Documents/ArUco_Detect/jvpkg && jevois-jvpkg ../ARCC_arucodetect.jvpkg

jvpkg: CMakeFiles/jvpkg
jvpkg: CMakeFiles/jvpkg.dir/build.make

.PHONY : jvpkg

# Rule to build all files generated by this target.
CMakeFiles/jvpkg.dir/build: jvpkg

.PHONY : CMakeFiles/jvpkg.dir/build

CMakeFiles/jvpkg.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/jvpkg.dir/cmake_clean.cmake
.PHONY : CMakeFiles/jvpkg.dir/clean

CMakeFiles/jvpkg.dir/depend:
	cd /home/jake/Documents/ArUco_Detect/pbuild && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jake/Documents/ArUco_Detect /home/jake/Documents/ArUco_Detect /home/jake/Documents/ArUco_Detect/pbuild /home/jake/Documents/ArUco_Detect/pbuild /home/jake/Documents/ArUco_Detect/pbuild/CMakeFiles/jvpkg.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/jvpkg.dir/depend

