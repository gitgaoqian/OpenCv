cmake_minimum_required(VERSION 2.8)
project( face_detect )
find_package( OpenCV REQUIRED )
add_executable( face_detect face_detect.cpp )
target_link_libraries( face_detect ${OpenCV_LIBS} )
