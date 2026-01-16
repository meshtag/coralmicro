# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(POSE_STREAMING_LINK_LIBRARIES
    libs_base-m7_freertos
    libs_base-m7_http_server
    libs_jpeg_m7
)

# USB (default target name kept for compatibility)
add_executable_m7(pose_streaming_http
    pose_streaming_http.cc
    DATA
    ${PROJECT_SOURCE_DIR}/models/pose_int8_edgetpu.tflite
    ${PROJECT_SOURCE_DIR}/examples/camera_streaming_http/web/coral_micro_camera.html>/coral_micro_camera.html
)

target_link_libraries(pose_streaming_http
    ${POSE_STREAMING_LINK_LIBRARIES}
)

# USB (explicit subapp)
add_executable_m7(pose_streaming_http_usb
    pose_streaming_http.cc
    DATA
    ${PROJECT_SOURCE_DIR}/models/pose_int8_edgetpu.tflite
    ${PROJECT_SOURCE_DIR}/examples/camera_streaming_http/web/coral_micro_camera.html>/coral_micro_camera.html
)

target_link_libraries(pose_streaming_http_usb
    ${POSE_STREAMING_LINK_LIBRARIES}
)

# Ethernet
add_executable_m7(pose_streaming_http_ethernet
    pose_streaming_http.cc
    DATA
    ${PROJECT_SOURCE_DIR}/models/pose_int8_edgetpu.tflite
    ${PROJECT_SOURCE_DIR}/examples/camera_streaming_http/web/coral_micro_camera.html>/coral_micro_camera.html
)

target_compile_definitions(pose_streaming_http_ethernet PUBLIC
    CAMERA_STREAMING_HTTP_ETHERNET
)

target_link_libraries(pose_streaming_http_ethernet
    ${POSE_STREAMING_LINK_LIBRARIES}
    libs_base-m7_ethernet
)

# Wi-Fi
add_executable_m7(pose_streaming_http_wifi
    pose_streaming_http.cc
    DATA
    ${PROJECT_SOURCE_DIR}/models/pose_int8_edgetpu.tflite
    ${PROJECT_SOURCE_DIR}/examples/camera_streaming_http/web/coral_micro_camera.html>/coral_micro_camera.html
)

target_compile_definitions(pose_streaming_http_wifi PUBLIC
    CAMERA_STREAMING_HTTP_WIFI
)

target_link_libraries(pose_streaming_http_wifi
    ${POSE_STREAMING_LINK_LIBRARIES}
    libs_nxp_rt1176-sdk_wiced
    libs_base-m7_wifi
)
