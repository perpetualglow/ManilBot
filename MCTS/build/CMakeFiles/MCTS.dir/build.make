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

# Include any dependencies generated for this target.
include CMakeFiles/MCTS.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/MCTS.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MCTS.dir/flags.make

CMakeFiles/MCTS.dir/main.cpp.o: CMakeFiles/MCTS.dir/flags.make
CMakeFiles/MCTS.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frigoow/Documents/Manil/ManilAI/MCTS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MCTS.dir/main.cpp.o"
	/bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MCTS.dir/main.cpp.o -c /home/frigoow/Documents/Manil/ManilAI/MCTS/main.cpp

CMakeFiles/MCTS.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MCTS.dir/main.cpp.i"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/frigoow/Documents/Manil/ManilAI/MCTS/main.cpp > CMakeFiles/MCTS.dir/main.cpp.i

CMakeFiles/MCTS.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MCTS.dir/main.cpp.s"
	/bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/frigoow/Documents/Manil/ManilAI/MCTS/main.cpp -o CMakeFiles/MCTS.dir/main.cpp.s

# Object files for target MCTS
MCTS_OBJECTS = \
"CMakeFiles/MCTS.dir/main.cpp.o"

# External object files for target MCTS
MCTS_EXTERNAL_OBJECTS =

MCTS: CMakeFiles/MCTS.dir/main.cpp.o
MCTS: CMakeFiles/MCTS.dir/build.make
MCTS: CMakeFiles/MCTS.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/frigoow/Documents/Manil/ManilAI/MCTS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable MCTS"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MCTS.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MCTS.dir/build: MCTS

.PHONY : CMakeFiles/MCTS.dir/build

CMakeFiles/MCTS.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MCTS.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MCTS.dir/clean

CMakeFiles/MCTS.dir/depend:
	cd /home/frigoow/Documents/Manil/ManilAI/MCTS/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frigoow/Documents/Manil/ManilAI/MCTS /home/frigoow/Documents/Manil/ManilAI/MCTS /home/frigoow/Documents/Manil/ManilAI/MCTS/build /home/frigoow/Documents/Manil/ManilAI/MCTS/build /home/frigoow/Documents/Manil/ManilAI/MCTS/build/CMakeFiles/MCTS.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MCTS.dir/depend

