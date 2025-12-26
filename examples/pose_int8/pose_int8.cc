// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/camera/camera.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace coralmicro {
namespace {

constexpr char kModelPath[] = "/models/pose_int8_edgetpu.tflite";
// Arena sized generously; the EdgeTPU custom op consumes most of the buffers.
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);
constexpr float kHeatmapThreshold = 0.01f;
constexpr float kPafThreshold = 0.01f;
constexpr int kNumKeypoints = 18;  // background channel is ignored
constexpr int kPointsPerLimb = 10;
constexpr int kPoseEntrySize = 20;
constexpr int kSuppressionRadius = 6;

constexpr int kBodyPartsKptIds[19][2] = {
    {1, 2},  {1, 5},  {2, 3},  {3, 4},  {5, 6},  {6, 7},  {1, 8},
    {8, 9},  {9, 10}, {1, 11}, {11, 12},{12, 13},{1, 0},  {0, 14},
    {14, 16},{0, 15}, {15, 17},{2, 16}, {5, 17}};
constexpr int kBodyPartsPafIds[19][2] = {
    {12, 13}, {20, 21}, {14, 15}, {16, 17}, {22, 23}, {24, 25}, {0, 1},
    {2, 3},   {4, 5},   {6, 7},   {8, 9},   {10, 11}, {28, 29}, {30, 31},
    {34, 35}, {32, 33}, {36, 37}, {18, 19}, {26, 27}};
constexpr const char* kKeypointNames[kNumKeypoints] = {
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder",
    "LElbow", "LWrist", "RHip", "RKnee", "RAnkle", "LHip", "LKnee",
    "LAnkle", "REye", "LEye", "REar", "LEar"};

struct Peak {
  int x;
  int y;
  float score;
  int id;
};

struct Pose {
  int keypoints[kNumKeypoints][2];  // -1 if not present
  float confidence;
  int found;
};

struct TensorStats {
  float min_v;
  float max_v;
};

TensorStats ComputeTensorStats(const TfLiteTensor* tensor) {
  TensorStats stats{std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity()};
  const auto* data = tflite::GetTensorData<int8_t>(tensor);
  for (int i = 0; i < tensor->bytes; ++i) {
    const float v = (data[i] - tensor->params.zero_point) * tensor->params.scale;
    if (v < stats.min_v) stats.min_v = v;
    if (v > stats.max_v) stats.max_v = v;
  }
  return stats;
}

float DequantValue(const TfLiteTensor* tensor, int idx) {
  const auto* data = tflite::GetTensorData<int8_t>(tensor);
  return (data[idx] - tensor->params.zero_point) * tensor->params.scale;
}

std::vector<float> DequantTensor(const TfLiteTensor* tensor) {
  const int height = tensor->dims->data[1];
  const int width = tensor->dims->data[2];
  const int channels = tensor->dims->data[3];
  std::vector<float> out(height * width * channels);
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = DequantValue(tensor, static_cast<int>(i));
  }
  return out;
}

inline float At(const std::vector<float>& data, int h, int w, int c,
                int height, int width, int channels) {
  return data[(h * width + w) * channels + c];
}

float MaxChannelDequant(const TfLiteTensor* tensor, int channel) {
  const int height = tensor->dims->data[1];
  const int width = tensor->dims->data[2];
  const int channels = tensor->dims->data[3];
  const auto* data = tflite::GetTensorData<int8_t>(tensor);
  float max_v = -std::numeric_limits<float>::infinity();
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int idx = (y * width + x) * channels + channel;
      float v = (data[idx] - tensor->params.zero_point) * tensor->params.scale;
      if (v > max_v) max_v = v;
    }
  }
  return max_v;
}

std::vector<float> ResizeBilinear(const std::vector<float>& src, int in_h,
                                  int in_w, int channels, int out_h,
                                  int out_w) {
  std::vector<float> dst(out_h * out_w * channels);
  const float scale_y = static_cast<float>(in_h) / static_cast<float>(out_h);
  const float scale_x = static_cast<float>(in_w) / static_cast<float>(out_w);
  for (int y = 0; y < out_h; ++y) {
    float fy = (y + 0.5f) * scale_y - 0.5f;
    int y0 = static_cast<int>(std::floor(fy));
    int y1 = std::min(y0 + 1, in_h - 1);
    float wy1 = fy - y0;
    float wy0 = 1.0f - wy1;
    y0 = std::max(0, y0);
    for (int x = 0; x < out_w; ++x) {
      float fx = (x + 0.5f) * scale_x - 0.5f;
      int x0 = static_cast<int>(std::floor(fx));
      int x1 = std::min(x0 + 1, in_w - 1);
      float wx1 = fx - x0;
      float wx0 = 1.0f - wx1;
      x0 = std::max(0, x0);
      for (int c = 0; c < channels; ++c) {
        float v00 = At(src, y0, x0, c, in_h, in_w, channels);
        float v01 = At(src, y0, x1, c, in_h, in_w, channels);
        float v10 = At(src, y1, x0, c, in_h, in_w, channels);
        float v11 = At(src, y1, x1, c, in_h, in_w, channels);
        float v0 = v00 * wx0 + v01 * wx1;
        float v1 = v10 * wx0 + v11 * wx1;
        float v = v0 * wy0 + v1 * wy1;
        dst[(y * out_w + x) * channels + c] = v;
      }
    }
  }
  return dst;
}

std::vector<Peak> ExtractKeypoints(const std::vector<float>& heatmap,
                                   int height, int width, int channel,
                                   int channels, int& total_kpt_num) {
  std::vector<Peak> kpts;
  // Collect local maxima above threshold.
  struct Candidate {
    int x;
    int y;
    float v;
  };
  std::vector<Candidate> candidates;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float v = At(heatmap, y, x, channel, height, width, channels);
      if (v < kHeatmapThreshold) continue;
      const float left = (x > 0) ? At(heatmap, y, x - 1, channel, height, width, channels) : 0.f;
      const float right = (x + 1 < width) ? At(heatmap, y, x + 1, channel, height, width, channels) : 0.f;
      const float up = (y > 0) ? At(heatmap, y - 1, x, channel, height, width, channels) : 0.f;
      const float down = (y + 1 < height) ? At(heatmap, y + 1, x, channel, height, width, channels) : 0.f;
      if (v > left && v > right && v > up && v > down) {
        candidates.push_back({x, y, v});
      }
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.x < b.x; });
  std::vector<uint8_t> suppressed(candidates.size(), 0);
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (suppressed[i]) continue;
    const auto& ci = candidates[i];
    for (size_t j = i + 1; j < candidates.size(); ++j) {
      if (suppressed[j]) continue;
      const auto& cj = candidates[j];
      const float dx = static_cast<float>(ci.x - cj.x);
      const float dy = static_cast<float>(ci.y - cj.y);
      if (std::sqrt(dx * dx + dy * dy) < kSuppressionRadius) {
        suppressed[j] = 1;
      }
    }
    kpts.push_back({ci.x, ci.y, ci.v, total_kpt_num++});
  }
  return kpts;
}

// Greedy NMS on limb connections to enforce unique start/end per limb.
void ConnectionsNms(std::vector<int>& a_idx, std::vector<int>& b_idx,
                    std::vector<float>& scores) {
  const size_t n = scores.size();
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(),
            [&](size_t i, size_t j) { return scores[i] > scores[j]; });
  std::vector<int> chosen_a;
  std::vector<int> chosen_b;
  std::vector<int> a_out;
  std::vector<int> b_out;
  std::vector<float> s_out;
  for (size_t idx : order) {
    if (std::find(chosen_a.begin(), chosen_a.end(), a_idx[idx]) != chosen_a.end()) continue;
    if (std::find(chosen_b.begin(), chosen_b.end(), b_idx[idx]) != chosen_b.end()) continue;
    chosen_a.push_back(a_idx[idx]);
    chosen_b.push_back(b_idx[idx]);
    a_out.push_back(a_idx[idx]);
    b_out.push_back(b_idx[idx]);
    s_out.push_back(scores[idx]);
  }
  a_idx.swap(a_out);
  b_idx.swap(b_out);
  scores.swap(s_out);
}

std::vector<std::vector<float>> GroupKeypoints(
    const std::vector<std::vector<Peak>>& all_kpts_by_type,
    const std::vector<Peak>& all_kpts_flat, const std::vector<float>& pafs,
    int height, int width) {
  std::vector<std::vector<float>> pose_entries;
  const int channels = 38;

  for (int part_id = 0; part_id < 19; ++part_id) {
    const auto& kpts_a = all_kpts_by_type[kBodyPartsKptIds[part_id][0]];
    const auto& kpts_b = all_kpts_by_type[kBodyPartsKptIds[part_id][1]];
    if (kpts_a.empty() || kpts_b.empty()) continue;

    std::vector<int> a_idx_list;
    std::vector<int> b_idx_list;
    std::vector<float> affinity_list;

    for (size_t i = 0; i < kpts_a.size(); ++i) {
      for (size_t j = 0; j < kpts_b.size(); ++j) {
        const float vec_x = static_cast<float>(kpts_b[j].x - kpts_a[i].x);
        const float vec_y = static_cast<float>(kpts_b[j].y - kpts_a[i].y);
        const float norm = std::sqrt(vec_x * vec_x + vec_y * vec_y);
        if (norm < 1e-6f) continue;
        const float vx = vec_x / norm;
        const float vy = vec_y / norm;
        float score_sum = 0.f;
        int valid_points = 0;
        for (int p = 0; p < kPointsPerLimb; ++p) {
          const float alpha = static_cast<float>(p) / (kPointsPerLimb - 1);
          int px = static_cast<int>(std::round(kpts_a[i].x + alpha * vec_x));
          int py = static_cast<int>(std::round(kpts_a[i].y + alpha * vec_y));
          px = std::max(0, std::min(width - 1, px));
          py = std::max(0, std::min(height - 1, py));
          const int c1 = kBodyPartsPafIds[part_id][0];
          const int c2 = kBodyPartsPafIds[part_id][1];
          const float paf_x = At(pafs, py, px, c1, height, width, channels);
          const float paf_y = At(pafs, py, px, c2, height, width, channels);
          const float affinity = paf_x * vx + paf_y * vy;
        if (affinity > kPafThreshold) {
          score_sum += affinity;
          ++valid_points;
        }
      }
        const float success_ratio =
            static_cast<float>(valid_points) / static_cast<float>(kPointsPerLimb);
        if (valid_points > 0 && success_ratio > 0.5f) {
          const float mean_affinity = score_sum / valid_points;
          if (mean_affinity > 0.f) {
            a_idx_list.push_back(kpts_a[i].id);
            b_idx_list.push_back(kpts_b[j].id);
            affinity_list.push_back(mean_affinity);
          }
        }
      }
    }

    if (affinity_list.empty()) continue;
    ConnectionsNms(a_idx_list, b_idx_list, affinity_list);
    std::vector<std::tuple<int, int, float>> connections;
    for (size_t i = 0; i < affinity_list.size(); ++i) {
      connections.emplace_back(a_idx_list[i], b_idx_list[i], affinity_list[i]);
    }

    if (part_id == 0) {
      pose_entries.reserve(connections.size());
      for (const auto& c : connections) {
        std::vector<float> entry(kPoseEntrySize, -1.f);
        entry[kBodyPartsKptIds[part_id][0]] = static_cast<float>(std::get<0>(c));
        entry[kBodyPartsKptIds[part_id][1]] = static_cast<float>(std::get<1>(c));
        entry[kPoseEntrySize - 1] = 2.f;  // num keypoints
        entry[kPoseEntrySize - 2] =
            all_kpts_flat[std::get<0>(c)].score + all_kpts_flat[std::get<1>(c)].score +
            std::get<2>(c);
        pose_entries.push_back(entry);
      }
    } else if (part_id == 17 || part_id == 18) {
      // Ears connect to eyes.
      const int kpt_a_id = kBodyPartsKptIds[part_id][0];
      const int kpt_b_id = kBodyPartsKptIds[part_id][1];
      for (const auto& c : connections) {
        for (auto& entry : pose_entries) {
          if (entry[kpt_a_id] == std::get<0>(c) && entry[kpt_b_id] == -1.f) {
            entry[kpt_b_id] = std::get<1>(c);
          } else if (entry[kpt_b_id] == std::get<1>(c) && entry[kpt_a_id] == -1.f) {
            entry[kpt_a_id] = std::get<0>(c);
          }
        }
      }
    } else {
      const int kpt_a_id = kBodyPartsKptIds[part_id][0];
      const int kpt_b_id = kBodyPartsKptIds[part_id][1];
      for (const auto& c : connections) {
        int num_assigned = 0;
        for (auto& entry : pose_entries) {
          if (entry[kpt_a_id] == std::get<0>(c) && entry[kpt_b_id] == -1.f) {
            entry[kpt_b_id] = std::get<1>(c);
            entry[kPoseEntrySize - 1] += 1.f;
            entry[kPoseEntrySize - 2] +=
                all_kpts_flat[std::get<1>(c)].score + std::get<2>(c);
            num_assigned++;
          }
        }
        if (num_assigned == 0) {
          std::vector<float> entry(kPoseEntrySize, -1.f);
          entry[kpt_a_id] = std::get<0>(c);
          entry[kpt_b_id] = std::get<1>(c);
          entry[kPoseEntrySize - 1] = 2.f;
          entry[kPoseEntrySize - 2] =
              all_kpts_flat[std::get<0>(c)].score +
              all_kpts_flat[std::get<1>(c)].score + std::get<2>(c);
          pose_entries.push_back(entry);
        }
      }
    }
  }

  // Filter low-quality poses.
  std::vector<std::vector<float>> filtered;
  for (const auto& entry : pose_entries) {
    if (entry[kPoseEntrySize - 1] < 3.f) continue;
    if ((entry[kPoseEntrySize - 2] / entry[kPoseEntrySize - 1]) < 0.2f) continue;
    filtered.push_back(entry);
  }
  return filtered;
}

std::vector<Pose> DecodePoses(const TfLiteTensor* heatmap_a,
                              const TfLiteTensor* heatmap_b,
                              const TfLiteTensor* paf_a,
                              const TfLiteTensor* paf_b,
                              int input_height, int input_width,
                              int upsample_factor) {
  std::vector<Pose> poses;
  if (!heatmap_a || !paf_a) return poses;

  // Combine dual heads (if present) by elementwise max.
  const int hm_h = heatmap_a->dims->data[1];
  const int hm_w = heatmap_a->dims->data[2];
  const int hm_c = heatmap_a->dims->data[3];
  const int paf_h = paf_a->dims->data[1];
  const int paf_w = paf_a->dims->data[2];
  const int paf_c = paf_a->dims->data[3];

  auto hm0 = DequantTensor(heatmap_a);
  std::vector<float> heatmaps = hm0;
  if (heatmap_b) {
    auto hm1 = DequantTensor(heatmap_b);
    for (size_t i = 0; i < heatmaps.size(); ++i) {
      heatmaps[i] = std::max(heatmaps[i], hm1[i]);
    }
  }

  auto paf0 = DequantTensor(paf_a);
  std::vector<float> pafs = paf0;
  if (paf_b) {
    auto paf1 = DequantTensor(paf_b);
    for (size_t i = 0; i < pafs.size(); ++i) {
      pafs[i] = std::max(pafs[i], paf1[i]);
    }
  }

  const int up_factor = std::max(1, upsample_factor);
  int up_hm_h = hm_h;
  int up_hm_w = hm_w;
  int up_paf_h = paf_h;
  int up_paf_w = paf_w;
  if (up_factor > 1) {
    up_hm_h = hm_h * up_factor;
    up_hm_w = hm_w * up_factor;
    up_paf_h = paf_h * up_factor;
    up_paf_w = paf_w * up_factor;
    heatmaps = ResizeBilinear(heatmaps, hm_h, hm_w, hm_c, up_hm_h, up_hm_w);
    pafs = ResizeBilinear(pafs, paf_h, paf_w, paf_c, up_paf_h, up_paf_w);
  }

  int total_kpt_num = 0;
  std::vector<std::vector<Peak>> all_kpts_by_type(kNumKeypoints);
  std::vector<Peak> all_kpts_flat;
  for (int kpt_idx = 0; kpt_idx < kNumKeypoints; ++kpt_idx) {
    auto kpts = ExtractKeypoints(heatmaps, up_hm_h, up_hm_w, kpt_idx, hm_c,
                                 total_kpt_num);
    for (const auto& k : kpts) all_kpts_flat.push_back(k);
    all_kpts_by_type[kpt_idx] = std::move(kpts);
  }

  auto pose_entries = GroupKeypoints(all_kpts_by_type, all_kpts_flat, pafs,
                                     up_paf_h, up_paf_w);

  const float stride_h = static_cast<float>(input_height) / static_cast<float>(up_hm_h);
  const float stride_w = static_cast<float>(input_width) / static_cast<float>(up_hm_w);
  for (const auto& entry : pose_entries) {
    Pose pose{};
    pose.confidence = entry[kPoseEntrySize - 2];
    pose.found = static_cast<int>(entry[kPoseEntrySize - 1]);
    for (int k = 0; k < kNumKeypoints; ++k) {
      pose.keypoints[k][0] = -1;
      pose.keypoints[k][1] = -1;
      if (entry[k] < 0) continue;
      const int kid = static_cast<int>(entry[k]);
      const auto& kp = all_kpts_flat[kid];
      pose.keypoints[k][0] = static_cast<int>(kp.x * stride_w);
      pose.keypoints[k][1] = static_cast<int>(kp.y * stride_h);
    }
    poses.push_back(pose);
  }
  return poses;
}

[[noreturn]] void Main() {
  printf("Pose (int8 EdgeTPU) demo\r\n");
  LedSet(Led::kStatus, true);

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }

  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    printf("ERROR: Failed to get EdgeTPU context\r\n");
    vTaskSuspend(nullptr);
  }

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<1> resolver;
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena, kTensorArenaSize,
                                       &error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }

  if (interpreter.inputs().size() != 1) {
    printf("ERROR: Model must have only one input tensor\r\n");
    vTaskSuspend(nullptr);
  }

  auto* input = interpreter.input_tensor(0);
  if (input->type != kTfLiteInt8) {
    printf("ERROR: Expected int8 input tensor\r\n");
    vTaskSuspend(nullptr);
  }

  const int model_height = input->dims->data[1];
  const int model_width = input->dims->data[2];
  const int model_channels = input->dims->data[3];
  printf("Input: %dx%dx%d (HxWxC) scale=%f zp=%ld\r\n", model_height,
         model_width, model_channels, input->params.scale,
         static_cast<long>(input->params.zero_point));

  // Capture into a uint8 buffer, then shift to int8 (zero-point = 0).
  std::vector<uint8_t> camera_buffer(input->bytes);

  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);

  // Locate outputs: two heatmap heads (19 channels incl. background) and two
  // PAF heads (38 channels).
  const TfLiteTensor* heatmaps[2] = {nullptr, nullptr};
  const TfLiteTensor* pafs[2] = {nullptr, nullptr};
  size_t hm_found = 0;
  size_t paf_found = 0;
  for (size_t i = 0; i < interpreter.outputs().size(); ++i) {
    const auto* t = interpreter.output_tensor(i);
    if (t->dims->size != 4) continue;
    if (t->dims->data[3] == 19 && hm_found < 2) {
      heatmaps[hm_found++] = t;
    } else if (t->dims->data[3] == 38 && paf_found < 2) {
      pafs[paf_found++] = t;
    }
  }
  if (!heatmaps[0] || !pafs[0]) {
    printf("ERROR: Did not find expected heatmap/paf outputs\r\n");
    vTaskSuspend(nullptr);
  }

  while (true) {
    CameraFrameFormat fmt{
        /*fmt=*/CameraFormat::kRgb,
        /*filter=*/CameraFilterMethod::kBilinear,
        /*rotation=*/CameraRotation::k270,
        /*width=*/model_width,
        /*height=*/model_height,
        /*preserve_ratio=*/false,
        /*buffer=*/camera_buffer.data()};
    if (!CameraTask::GetSingleton()->GetFrame({fmt})) {
      printf("Failed to capture image\r\n");
      vTaskSuspend(nullptr);
    }

    auto* input_data = tflite::GetTensorData<int8_t>(input);
    // Convert RGB camera buffer to BGR (model was trained on OpenCV BGR) and
    // zero-center with mean 128.
    for (size_t i = 0; i + 2 < camera_buffer.size(); i += 3) {
      const uint8_t r = camera_buffer[i + 0];
      const uint8_t g = camera_buffer[i + 1];
      const uint8_t b = camera_buffer[i + 2];
      input_data[i + 0] = static_cast<int8_t>(static_cast<int>(b) - 128);
      input_data[i + 1] = static_cast<int8_t>(static_cast<int>(g) - 128);
      input_data[i + 2] = static_cast<int8_t>(static_cast<int>(r) - 128);
    }

    if (interpreter.Invoke() != kTfLiteOk) {
      printf("Failed to invoke\r\n");
      vTaskSuspend(nullptr);
    }

    static int frame_counter = 0;
    if ((frame_counter++ % 10) == 0) {
      auto in_stats = ComputeTensorStats(input);
      float hm0 = MaxChannelDequant(heatmaps[0], 0);
      float hm1 = MaxChannelDequant(heatmaps[0], 1);
      float hm_all = MaxChannelDequant(heatmaps[0], 2);  // Shoulder as a proxy
      printf("Debug input: min=%.3f max=%.3f heatmax: nose=%.3f neck=%.3f sh=%.3f\r\n",
             in_stats.min_v, in_stats.max_v, hm0, hm1, hm_all);
    }

    auto poses =
        DecodePoses(heatmaps[0], heatmaps[1], pafs[0], pafs[1], model_height,
                    model_width, /*upsample_factor=*/4);
    bool used_fallback = false;
    if (poses.empty()) {
      poses = DecodePoses(heatmaps[0], heatmaps[1], pafs[0], pafs[1],
                          model_height, model_width, /*upsample_factor=*/1);
      used_fallback = !poses.empty();
    }
    static int dbg_counter = 0;
    if (poses.empty()) {
      printf("Poses: none\r\n");
      if ((dbg_counter++ % 10) == 0) {
        float hm0 = MaxChannelDequant(heatmaps[0], 0);
        float hm1 = MaxChannelDequant(heatmaps[0], 1);
      printf("  Debug heatmap max: Nose=%.3f Neck=%.3f\r\n", hm0, hm1);
    }
    } else {
      printf("Poses (%u)%s:\r\n", static_cast<unsigned int>(poses.size()),
             used_fallback ? " [fallback decode]" : "");
      for (size_t p = 0; p < poses.size(); ++p) {
        printf(" Pose %u conf=%.3f kpts=%d\r\n",
               static_cast<unsigned int>(p), poses[p].confidence,
               poses[p].found);
        for (int k = 0; k < kNumKeypoints; ++k) {
          if (poses[p].keypoints[k][0] < 0) continue;
          printf("  %-8s x=%3d y=%3d\r\n", kKeypointNames[k],
                 poses[p].keypoints[k][0], poses[p].keypoints[k][1]);
        }
      }
    }

    vTaskDelay(pdMS_TO_TICKS(100));
  }
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}
