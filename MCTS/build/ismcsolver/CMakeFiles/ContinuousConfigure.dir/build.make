# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/frigoow/Documents/Manil/ManilAI/MCTS

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/frigoow/Documents/Manil/ManilAI/MCTS/build

# Utility rule file for ContinuousConfigure.

# Include the progress variables for this target.
include ismcsolver/CMakeFiles/ContinuousConfigure.dir/progress.make

ismcsolver/CMakeFiles/ContinuousConfigure:
	cd /home/frigoow/Documents/Manil/ManilAI/MCTS/build/ismcsolver && /usr/bin/ctest -D ContinuousConfigure

ContinuousConfigure: ismcsolver/CMakeFiles/ContinuousConfigure
ContinuousConfigure: ismcsolver/CMakeFiles/ContinuousConfigure.dir/build.make

.PHONY : ContinuousConfigure

# Rule to build all files generated by this target.
ismcsolver/CMakeFiles/ContinuousConfigure.dir/build: ContinuousConfigure

.PHONY : ismcsolver/CMakeFiles/ContinuousConfigure.dir/build

ismcsolver/CMakeFiles/ContinuousConfigure.dir/clean:
	cd /home/frigoow/Documents/Manil/ManilAI/MCTS/build/ismcsolver && $(CMAKE_COMMAND) -P CMakeFiles/ContinuousConfigure.dir/cmake_clean.cmake
.PHONY : ismcsolver/CMakeFiles/ContinuousConfigure.dir/clean

ismcsolver/CMakeFiles/ContinuousConfigure.dir/depend:
	cd /home/frigoow/Documents/Manil/ManilAI/MCTS/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frigoow/Documents/Manil/ManilAI/MCTS /home/frigoow/Documents/Manil/ManilAI/MCTS/ismcsolver /home/frigoow/Documents/Manil/ManilAI/MCTS/build /home/frigoow/Documents/Manil/ManilAI/MCTS/build/ismcsolver /home/frigoow/Documents/Manil/ManilAI/MCTS/build/ismcsolver/CMakeFiles/ContinuousConfigure.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ismcsolver/CMakeFiles/ContinuousConfigure.dir/depend
