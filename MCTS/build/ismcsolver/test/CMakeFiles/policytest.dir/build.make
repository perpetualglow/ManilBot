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
include ismcsolver/test/CMakeFiles/policytest.dir/depend.make

# Include the progress variables for this target.
include ismcsolver/test/CMakeFiles/policytest.dir/progress.make

# Include the compile flags for this target's objects.
include ismcsolver/test/CMakeFiles/policytest.dir/flags.make

ismcsolver/test/CMakeFiles/policytest.dir/policytest.cpp.o: ismcsolver/test/CMakeFiles/policytest.dir/flags.make
ismcsolver/test/CMakeFiles/policytest.dir/policytest.cpp.o: ../ismcsolver/test/policytest.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/frigoow/Documents/Manil/ManilAI/MCTS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object ismcsolver/test/CMakeFiles/policytest.dir/policytest.cpp.o"
	cd /home/frigoow/Documents/Manil/ManilAI/MCTS/build/ismcsolver/test && /bin/g++-9  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/policytest.dir/policytest.cpp.o -c /home/frigoow/Documents/Manil/ManilAI/MCTS/ismcsolver/test/policytest.cpp

ismcsolver/test/CMakeFiles/policytest.dir/policytest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/policytest.dir/policytest.cpp.i"
	cd /home/frigoow/Documents/Manil/ManilAI/MCTS/build/ismcsolver/test && /bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/frigoow/Documents/Manil/ManilAI/MCTS/ismcsolver/test/policytest.cpp > CMakeFiles/policytest.dir/policytest.cpp.i

ismcsolver/test/CMakeFiles/policytest.dir/policytest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/policytest.dir/policytest.cpp.s"
	cd /home/frigoow/Documents/Manil/ManilAI/MCTS/build/ismcsolver/test && /bin/g++-9 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/frigoow/Documents/Manil/ManilAI/MCTS/ismcsolver/test/policytest.cpp -o CMakeFiles/policytest.dir/policytest.cpp.s

# Object files for target policytest
policytest_OBJECTS = \
"CMakeFiles/policytest.dir/policytest.cpp.o"

# External object files for target policytest
policytest_EXTERNAL_OBJECTS = \
"/home/frigoow/Documents/Manil/ManilAI/MCTS/build/ismcsolver/test/CMakeFiles/testmain.dir/main.cpp.o"

ismcsolver/test/policytest: ismcsolver/test/CMakeFiles/policytest.dir/policytest.cpp.o
ismcsolver/test/policytest: ismcsolver/test/CMakeFiles/testmain.dir/main.cpp.o
ismcsolver/test/policytest: ismcsolver/test/CMakeFiles/policytest.dir/build.make
ismcsolver/test/policytest: ismcsolver/test/CMakeFiles/policytest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/frigoow/Documents/Manil/ManilAI/MCTS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable policytest"
	cd /home/frigoow/Documents/Manil/ManilAI/MCTS/build/ismcsolver/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/policytest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ismcsolver/test/CMakeFiles/policytest.dir/build: ismcsolver/test/policytest

.PHONY : ismcsolver/test/CMakeFiles/policytest.dir/build

ismcsolver/test/CMakeFiles/policytest.dir/clean:
	cd /home/frigoow/Documents/Manil/ManilAI/MCTS/build/ismcsolver/test && $(CMAKE_COMMAND) -P CMakeFiles/policytest.dir/cmake_clean.cmake
.PHONY : ismcsolver/test/CMakeFiles/policytest.dir/clean

ismcsolver/test/CMakeFiles/policytest.dir/depend:
	cd /home/frigoow/Documents/Manil/ManilAI/MCTS/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/frigoow/Documents/Manil/ManilAI/MCTS /home/frigoow/Documents/Manil/ManilAI/MCTS/ismcsolver/test /home/frigoow/Documents/Manil/ManilAI/MCTS/build /home/frigoow/Documents/Manil/ManilAI/MCTS/build/ismcsolver/test /home/frigoow/Documents/Manil/ManilAI/MCTS/build/ismcsolver/test/CMakeFiles/policytest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ismcsolver/test/CMakeFiles/policytest.dir/depend
