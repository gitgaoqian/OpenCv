cmake_minimum_required(VERSION 2.8)
project( camera_read )
find_package( OpenCV REQUIRED )
add_executable( camera_read camera_read.cpp )
target_link_libraries( camera_read ${OpenCV_LIBS} )
