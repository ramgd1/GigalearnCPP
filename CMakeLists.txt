cmake_minimum_required (VERSION 3.10)

project("GigaLearnBot")

file(GLOB_RECURSE FILES_SRC "src/*.cpp" "src/*.h" "src/*.hpp")
add_executable(GigaLearnBot ${FILES_SRC})

if (MSVC)
	if (POLICY CMP0141)
		cmake_policy(SET CMP0141 NEW)
	endif()
	set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "Embedded")
	foreach(flag_var
		CMAKE_C_FLAGS
		CMAKE_C_FLAGS_DEBUG
		CMAKE_C_FLAGS_RELWITHDEBINFO
		CMAKE_CXX_FLAGS
		CMAKE_CXX_FLAGS_DEBUG
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
	)
		string(REPLACE "/Zi" "" ${flag_var} "${${flag_var}}")
		string(REPLACE "/wd9025" "" ${flag_var} "${${flag_var}}")
		set(${flag_var} "${${flag_var}}" CACHE STRING "" FORCE)
	endforeach()
endif()

# Set C++ version to 20
set_target_properties(GigaLearnBot PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(GigaLearnBot PROPERTIES CXX_STANDARD 20)

# Make sure GigaLearnCPP is going to build in the same directory as us
# Otherwise, we won't be able to import it at runtime
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Include RLGymSim_PPO
add_subdirectory(GigaLearnCPP)
target_link_libraries(GigaLearnBot GigaLearnCPP)

# Include RLBot
add_subdirectory(RLBotCPP)
target_link_libraries(GigaLearnBot RLBotCPP)
