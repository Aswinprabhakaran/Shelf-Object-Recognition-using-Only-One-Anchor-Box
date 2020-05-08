//
//  yolov3Detection.cpp
//  MNN
//
//  Created by Xiaobin Zhang on 2019/09/20.
//

#include <stdio.h>
#include "ImageProcess.hpp"
#include "Interpreter.hpp"
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <sys/time.h>
#include "AutoTime.hpp"
#include "ErrorCode.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace MNN;
using namespace MNN::CV;


// definition of a bbox prediction record
typedef struct prediction {
    float x;
    float y;
    float width;
    float height;
    float confidence;
    int class_index;
}t_prediction;


// model inference settings
struct Settings {
  int loop_count = 1;
  int number_of_threads = 4;
  int number_of_warmup_runs = 2;
  float input_mean = 0.0f;
  float input_std = 255.0f;
  std::string model_name = "./model.mnn";
  std::string input_img_name = "./dog.jpg";
  std::string classes_file_name = "./classes.txt";
  std::string anchors_file_name = "./yolo_anchors.txt";
  //bool verbose = false;
  //bool input_floating = false;
  //string input_layer_type = "uint8_t";
};


float sigmoid(float x)
{
    return (1 / (1 + exp(-x)));
}


double get_us(struct timeval t)
{
    return (t.tv_sec * 1000000 + t.tv_usec);
}


void display_usage() {
  std::cout
      << "Usage: yolov3Detection\n"
      << "--mnn_model, -m: model_name.mnn\n"
      << "--image, -i: image_name.jpg\n"
      << "--classes, -l: classes labels for the model\n"
      << "--anchors, -a: anchor values for the model\n"
      << "--input_mean, -b: input mean\n"
      << "--input_std, -s: input standard deviation\n"
      << "--threads, -t: number of threads\n"
      << "--count, -c: loop model run for certain times\n"
      << "--warmup_runs, -w: number of warmup runs\n"
      //<< "--verbose, -v: [0|1] print more information\n"
      << "\n";
}


// YOLO postprocess for each prediction feature map
void yolo_postprocess(const Tensor* feature_map, const int input_width, const int input_height,
                      const int num_classes, const std::vector<std::pair<float, float>> anchors,
                      std::vector<t_prediction> &prediction_list, float conf_threshold)
{
    // 1. do following transform to get the output bbox,
    //    which is aligned with YOLOv3 paper:
    //
    //    bbox_x = sigmoid(pred_x) + grid_w
    //    bbox_y = sigmoid(pred_y) + grid_h
    //    bbox_w = exp(pred_w) * anchor_w / stride
    //    bbox_h = exp(pred_h) * anchor_h / stride
    //    bbox_obj = sigmoid(pred_obj)
    //
    // 2. convert the grid scale coordinate back to
    //    input image shape, with stride:
    //
    //    bbox_x = bbox_x * stride;
    //    bbox_y = bbox_y * stride;
    //    bbox_w = bbox_w * stride;
    //    bbox_h = bbox_h * stride;
    //
    // 3. convert centoids to top left coordinates
    //
    //    bbox_x = bbox_x - (bbox_w / 2);
    //    bbox_y = bbox_y - (bbox_h / 2);
    //
    // 4. get bbox confidence (class_score * objectness)
    //    and filter with threshold
    //
    //    bbox_conf[:] = sigmoid(bbox_class_score[:]) * bbox_obj
    //    bbox_max_conf = max(bbox_conf[:])
    //    bbox_max_index = argmax(bbox_conf[:])
    //
    // 5. filter bbox_max_conf with threshold
    //
    //    if(bbox_max_conf > conf_threshold)
    //        enqueue the bbox info

    const float* data = feature_map->host<float>();
    auto dimType = feature_map->getDimensionType();

    auto batch   = feature_map->batch();
    auto channel = feature_map->channel();
    auto height  = feature_map->height();
    auto width   = feature_map->width();

    int stride = input_width / width;
    auto unit = sizeof(float);
    int anchor_num_per_layer = anchors.size();

    // now we only support single image postprocess
    MNN_ASSERT(batch == 1);

    // the featuremap channel should be like 3*(num_classes + 5)
    MNN_ASSERT(anchor_num_per_layer * (num_classes + 5) == channel);

    if (dimType == Tensor::TENSORFLOW) {
        // Tensorflow format tensor, NHWC
        MNN_PRINT("Tensorflow format: NHWC\n");

        auto bytesPerRow   = channel * unit;
        auto bytesPerImage = width * bytesPerRow;
        auto bytesPerBatch = height * bytesPerImage;

        for (int b = 0; b < batch; b++) {
            auto bytes = data + b * bytesPerBatch / unit;
            MNN_PRINT("batch %d:\n", b);

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    for (int anc = 0; anc < anchor_num_per_layer; anc++) {
                        //get bbox prediction data for each anchor, each feature point
                        int bbox_x_offset = h * width * channel + w * channel + anc * (num_classes + 5);
                        int bbox_y_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 1;
                        int bbox_w_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 2;
                        int bbox_h_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 3;
                        int bbox_obj_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 4;
                        int bbox_scores_offset = h * width * channel + w * channel + anc * (num_classes + 5) + 5;
                        int bbox_scores_step = 1;

                        float bbox_x = sigmoid(bytes[bbox_x_offset]) + w;
                        float bbox_y = sigmoid(bytes[bbox_y_offset]) + h;
                        float bbox_w = exp(bytes[bbox_w_offset]) * anchors[anc].first / stride;
                        float bbox_h = exp(bytes[bbox_h_offset]) * anchors[anc].second / stride;
                        float bbox_obj = sigmoid(bytes[bbox_obj_offset]);

                        // Transfer anchor coordinates
                        bbox_x = bbox_x * stride;
                        bbox_y = bbox_y * stride;
                        bbox_w = bbox_w * stride;
                        bbox_h = bbox_h * stride;

                        // Convert centoids to top left coordinates
                        bbox_x = bbox_x - (bbox_w / 2);
                        bbox_y = bbox_y - (bbox_h / 2);

                        //get anchor output confidence (class_score * objectness) and filter with threshold
                        float max_conf = 0.0;
                        int max_index = -1;
                        for (int i = 0; i < num_classes; i++) {
                            float tmp_conf = sigmoid(bytes[bbox_scores_offset + i * bbox_scores_step]) * bbox_obj;

                            if(tmp_conf > max_conf) {
                                max_conf = tmp_conf;
                                max_index = i;
                            }
                        }
                        if(max_conf >= conf_threshold) {
                            // got a valid prediction, form up data and push to result vector
                            t_prediction bbox_prediction;
                            bbox_prediction.x = bbox_x;
                            bbox_prediction.y = bbox_y;
                            bbox_prediction.width = bbox_w;
                            bbox_prediction.height = bbox_h;
                            bbox_prediction.confidence = max_conf;
                            bbox_prediction.class_index = max_index;

                            prediction_list.emplace_back(bbox_prediction);
                        }
                    }
                }
            }
        }
    } else if (dimType == Tensor::CAFFE) {
        // Caffe format tensor, NCHW
        MNN_PRINT("Caffe format: NCHW\n");
        auto bytesPerRow   = width * unit;
        auto bytesPerImage = height * bytesPerRow;
        auto bytesPerBatch = channel * bytesPerImage;

        for (int b = 0; b < batch; b++) {
            auto bytes = data + b * bytesPerBatch / unit;
            MNN_PRINT("batch %d:\n", b);

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    for (int anc = 0; anc < anchor_num_per_layer; anc++) {
                        //get bbox prediction data for each anchor, each feature point
                        int bbox_x_offset = anc * (num_classes + 5) * width * height + h * width + w;
                        int bbox_y_offset = (anc * (num_classes + 5) + 1) * width * height + h * width + w;
                        int bbox_w_offset = (anc * (num_classes + 5) + 2) * width * height + h * width + w;
                        int bbox_h_offset = (anc * (num_classes + 5) + 3) * width * height + h * width + w;
                        int bbox_obj_offset = (anc * (num_classes + 5) + 4) * width * height + h * width + w;
                        int bbox_scores_offset = (anc * (num_classes + 5) + 5) * width * height + h * width + w;
                        int bbox_scores_step = width * height;

                        float bbox_x = sigmoid(bytes[bbox_x_offset]) + w;
                        float bbox_y = sigmoid(bytes[bbox_y_offset]) + h;
                        float bbox_w = exp(bytes[bbox_w_offset]) * anchors[anc].first / stride;
                        float bbox_h = exp(bytes[bbox_h_offset]) * anchors[anc].second / stride;
                        float bbox_obj = sigmoid(bytes[bbox_obj_offset]);

                        // Transfer anchor coordinates
                        bbox_x = bbox_x * stride;
                        bbox_y = bbox_y * stride;
                        bbox_w = bbox_w * stride;
                        bbox_h = bbox_h * stride;

                        // Convert centoids to top left coordinates
                        bbox_x = bbox_x - (bbox_w / 2);
                        bbox_y = bbox_y - (bbox_h / 2);

                        //get anchor output confidence (class_score * objectness) and filter with threshold
                        float max_conf = 0.0;
                        int max_index = -1;
                        for (int i = 0; i < num_classes; i++) {
                            float tmp_conf = sigmoid(bytes[bbox_scores_offset + i * bbox_scores_step]) * bbox_obj;

                            if(tmp_conf > max_conf) {
                                max_conf = tmp_conf;
                                max_index = i;
                            }
                        }
                        if(max_conf >= conf_threshold) {
                            // got a valid prediction, form up data and push to result vector
                            t_prediction bbox_prediction;
                            bbox_prediction.x = bbox_x;
                            bbox_prediction.y = bbox_y;
                            bbox_prediction.width = bbox_w;
                            bbox_prediction.height = bbox_h;
                            bbox_prediction.confidence = max_conf;
                            bbox_prediction.class_index = max_index;

                            prediction_list.emplace_back(bbox_prediction);
                        }
                    }
                }
            }
        }

    } else if (dimType == Tensor::CAFFE_C4) {
        MNN_PRINT("Caffe format: NC4HW4, not supported\n");
        exit(-1);
    } else {
        MNN_PRINT("Invalid tensor dim type: %d\n", dimType);
        exit(-1);
    }
}


//calculate IoU for 2 prediction boxes
float get_iou(t_prediction pred1, t_prediction pred2)
{
    // area for box 1
    float x1min = pred1.x;
    float x1max = pred1.x + pred1.width;
    float y1min = pred1.y;
    float y1max = pred1.y + pred1.height;
    float area1 = pred1.width * pred1.height;

    // area for box 2
    float x2min = pred2.x;
    float x2max = pred2.x + pred2.width;
    float y2min = pred2.y;
    float y2max = pred2.y + pred2.height;
    float area2 = pred2.width * pred2.height;

    // get area for intersection box
    float x_inter_min = std::max(x1min, x2min);
    float x_inter_max = std::min(x1max, x2max);
    float y_inter_min = std::max(y1min, y2min);
    float y_inter_max = std::min(y1max, y2max);

    float width_inter = std::max(0.0f, x_inter_max - x_inter_min + 1);
    float height_inter = std::max(0.0f, y_inter_max - y_inter_min + 1);
    float area_inter = width_inter * height_inter;

    // return IoU
    return area_inter / (area1 + area2 - area_inter);
}


//ascend order sort for prediction records
bool compare_conf(t_prediction lpred, t_prediction rpred)
{
    if (lpred.confidence < rpred.confidence)
        return true;
    else
        return false;
}


// NMS operation for the prediction list
void nms_boxes(const std::vector<t_prediction> prediction_list, std::vector<t_prediction>& prediction_nms_list, int num_classes, float iou_threshold)
{
    MNN_PRINT("prediction_list size before NMS: %lu\n", prediction_list.size());
    //go through every class
    for (int i = 0; i < num_classes; i++) {

        //get prediction list for class i
        std::vector<t_prediction> class_pred_list;
        for (int j = 0; j < prediction_list.size(); j++) {
            if (prediction_list[j].class_index == i) {
                class_pred_list.emplace_back(prediction_list[j]);
            }
        }

        if(!class_pred_list.empty()) {
            std::vector<t_prediction> class_pick_list;
            // ascend sort the class prediction list
            std::sort(class_pred_list.begin(), class_pred_list.end(), compare_conf);

            while(class_pred_list.size() > 0) {
                // pick the max score prediction result
                t_prediction current_pred = class_pred_list.back();
                class_pick_list.emplace_back(current_pred);
                class_pred_list.pop_back();

                // loop the list to get IoU with max score prediction
                for(auto iter = class_pred_list.begin(); iter != class_pred_list.end();) {
                    float iou = get_iou(current_pred, *iter);
                    // drop if IoU is larger than threshold
                    if(iou > iou_threshold) {
                        iter = class_pred_list.erase(iter);
                    } else {
                        iter++;
                    }
                }
            }

            // merge the picked predictions to final list
            prediction_nms_list.insert(prediction_nms_list.end(), class_pick_list.begin(), class_pick_list.end());
        }
    }
}


// select anchorset for corresponding featuremap layer
std::vector<std::pair<float, float>> get_anchorset(std::vector<std::pair<float, float>> anchors, const int feature_width, const int input_width)
{
    std::vector<std::pair<float, float>> anchorset;
    int anchor_num = anchors.size();

    // stride could confirm the feature map level:
    // image_input: 1 x 416 x 416 x 3
    // stride 32: 1 x 13 x 13 x 3 x (num_classes + 5)
    // stride 16: 1 x 26 x 26 x 3 x (num_classes + 5)
    // stride 8: 1 x 52 x 52 x 3 x (num_classes + 5)
    int stride = input_width / feature_width;

    // YOLOv3 model has 9 anchors and 3 feature layers
    if (anchor_num == 9) {
        if (stride == 32) {
            anchorset.emplace_back(anchors[6]);
            anchorset.emplace_back(anchors[7]);
            anchorset.emplace_back(anchors[8]);
        }
        else if (stride == 16) {
            anchorset.emplace_back(anchors[3]);
            anchorset.emplace_back(anchors[4]);
            anchorset.emplace_back(anchors[5]);
        }
        else if (stride == 8) {
            anchorset.emplace_back(anchors[0]);
            anchorset.emplace_back(anchors[1]);
            anchorset.emplace_back(anchors[2]);
        }
        else {
            MNN_PRINT("invalid feature map stride for anchorset!\n");
            exit(-1);
        }
    }
    // Tiny YOLOv3 model has 6 anchors and 2 feature layers
    else if (anchor_num == 6) {
        if (stride == 32) {
            anchorset.emplace_back(anchors[3]);
            anchorset.emplace_back(anchors[4]);
            anchorset.emplace_back(anchors[5]);
        }
        else if (stride == 16) {
            anchorset.emplace_back(anchors[0]);
            anchorset.emplace_back(anchors[1]);
            anchorset.emplace_back(anchors[2]);
        }
        else {
            MNN_PRINT("invalid anchorset index!\n");
            exit(-1);
        }
    }
    else {
        MNN_PRINT("invalid anchor numbers!\n");
        exit(-1);
    }

    return anchorset;
}


void parse_anchors(std::string line, std::vector<std::pair<float, float>>& anchors)
{
    // parse anchor definition txt file
    // which should be like follow:
    //
    // yolo_anchors:
    // 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
    //
    // tiny_yolo_anchors:
    // 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
    size_t curr = 0, next = 0;

    while(next != std::string::npos) {
        //get 1st number
        next = line.find(",", curr);
        std::string num1 = line.substr(curr, next-curr);
        //get 2nd number
        curr = next + 1;
        next = line.find(",  ", curr);
        std::string num2 = line.substr(curr, next-curr);
        //form up anchor
        anchors.emplace_back(std::make_pair(std::stoi(num1), std::stoi(num2)));
        //get start of next anchor
        curr = next + 3;
    }
}

void adjust_boxes(std::vector<t_prediction> &prediction_nms_list, int image_width, int image_height, int input_width, int input_height)
{
    // Rescale the final prediction back to original image
    float scale_width = (float)(image_width-1) / (input_width-1);
    float scale_height = (float)(image_height-1) / (input_height-1);

    for(auto &prediction_nms : prediction_nms_list) {
        prediction_nms.x = prediction_nms.x * scale_width;
        prediction_nms.y = prediction_nms.y * scale_height;
        prediction_nms.width = prediction_nms.width * scale_width;
        prediction_nms.height = prediction_nms.height * scale_height;
    }
}


void RunInference(Settings* s) {
    // record run time for every stage
    struct timeval start_time, stop_time;

    // create model & session
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(s->model_name.c_str()));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_AUTO;
    config.numThread = s->number_of_threads;
    auto session = net->createSession(config);

    // get input tensor info
    // assume only 1 input tensor (image_input)
    auto inputs = net->getSessionInputAll(session);
    MNN_ASSERT(inputs.size() == 1);
    auto image_input = inputs.begin()->second;

    auto shape = image_input->shape();
    int input_width = image_input->width();
    int input_height = image_input->height();
    int input_channel = image_input->channel();
    if (input_channel == 0)
        input_channel = 1;
    if (input_height == 0)
        input_height = 1;
    if (input_width == 0)
        input_width = 1;
    MNN_PRINT("image_input: width:%d , height:%d, channel: %d\n", input_width, input_height, input_channel);

    shape[0] = 1;
    net->resizeTensor(image_input, shape);
    net->resizeSession(session);

    // get output tensor info (e.g. for YOLOv3 arch):
    //image_input: 1 x 416 x 416 x 3
    //"conv2d_3/Conv2D": 1 x 13 x 13 x 3 x (num_classes + 5)
    //"conv2d_8/Conv2D": 1 x 26 x 26 x 3 x (num_classes + 5)
    //"conv2d_13/Conv2D": 1 x 52 x 52 x 3 x (num_classes + 5)
    auto outputs = net->getSessionOutputAll(session);
    int num_layers = outputs.size();

    // get classes labels
    std::vector<std::string> classes;
    std::ifstream classesOs(s->classes_file_name.c_str());
    std::string line;
    while (std::getline(classesOs, line)) {
        classes.emplace_back(line);
    }
    int num_classes = classes.size();
    MNN_PRINT("num_classes: %d\n", num_classes);

    // get anchor value
    std::vector<std::pair<float, float>> anchors;
    std::ifstream anchorsOs(s->anchors_file_name.c_str());
    while (std::getline(anchorsOs, line)) {
        parse_anchors(line, anchors);
    }

    // For YOLOv3 model, we should have 9 anchors and 3 feature layers
    // For Tiny YOLOv3 model, we should have 6 anchors and 2 feature layers
    MNN_ASSERT(anchors.size() / num_layers == 3);

    // load input image
    auto inputPatch = s->input_img_name.c_str();
    int image_width, image_height, image_channel;
    auto inputImage = stbi_load(inputPatch, &image_width, &image_height, &image_channel, 3);
    if (nullptr == inputImage) {
        MNN_ERROR("Can't open %s\n", inputPatch);
        return;
    }
    MNN_PRINT("origin image size: width:%d, height:%d, channel:%d\n", image_width, image_height, image_channel);
    Matrix trans;
    trans.setScale((float)(image_width-1) / (input_width-1), (float)(image_height-1) / (input_height-1));

    ImageProcess::Config pretreat_config;
    pretreat_config.filterType = BILINEAR;

    // normalize image input, In current training code we just do image/255.0
    // so default mean=0, std=255. Change it if you use a different normalization
    float input_normal = 1.0f / s->input_std;
    float mean[3]     = {s->input_mean, s->input_mean, s->input_mean};
    float normals[3] = {input_normal, input_normal, input_normal};

    ::memcpy(pretreat_config.mean, mean, sizeof(mean));
    ::memcpy(pretreat_config.normal, normals, sizeof(normals));
    pretreat_config.sourceFormat = RGB;
    pretreat_config.destFormat   = RGB;

    std::shared_ptr<ImageProcess> pretreat(ImageProcess::create(pretreat_config));
    pretreat->setMatrix(trans);
    pretreat->convert((uint8_t*)inputImage, image_width, image_height, 0, image_input);

    // run warm up session
    if (s->loop_count > 1)
        for (int i = 0; i < s->number_of_warmup_runs; i++) {
            pretreat->convert((uint8_t*)inputImage, image_width, image_height, 0, image_input);
            if (net->runSession(session) != NO_ERROR) {
                MNN_PRINT("Failed to invoke MNN!\n");
            }
        }

    // run model sessions to get output
    gettimeofday(&start_time, nullptr);
    for (int i = 0; i < s->loop_count; i++) {
        pretreat->convert((uint8_t*)inputImage, image_width, image_height, 0, image_input);
        if (net->runSession(session) != NO_ERROR) {
            MNN_PRINT("Failed to invoke MNN!\n");
        }
    }
    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("model invoke average time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / (1000 * s->loop_count));

    // free input image
    stbi_image_free(inputImage);

    // Copy output tensors to host, for further postprocess
    std::vector<std::shared_ptr<Tensor>> featureTensors;
    for(auto output : outputs) {
        MNN_PRINT("output tensor name: %s\n", output.first.c_str());
        auto output_tensor = output.second;
        auto dim_type = output_tensor->getDimensionType();
        if (output_tensor->getType().code != halide_type_float) {
            dim_type = Tensor::TENSORFLOW;
        }
        std::shared_ptr<Tensor> output_user(new Tensor(output_tensor, dim_type));
        output_tensor->copyToHostTensor(output_user.get());
        featureTensors.emplace_back(output_user);
    }

    // Do yolo_postprocess to parse out valid predictions
    std::vector<t_prediction> prediction_list;
    float conf_threshold = 0.1;
    float iou_threshold = 0.4;

    gettimeofday(&start_time, nullptr);

    for (int i = 0; i < num_layers; ++i) {
        Tensor* feature_map = featureTensors[i].get();
        std::vector<std::pair<float, float>> anchorset = get_anchorset(anchors, feature_map->width(), input_width);

        MNN_ASSERT(featureTensors[i]->getType().code == halide_type_float);
        yolo_postprocess(featureTensors[i].get(), input_width, input_height, num_classes, anchorset, prediction_list, conf_threshold);
    }

    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("yolo_postprocess time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / 1000);

    // Do NMS for predictions
    std::vector<t_prediction> prediction_nms_list;
    gettimeofday(&start_time, nullptr);
    nms_boxes(prediction_list, prediction_nms_list, num_classes, iou_threshold);
    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("NMS time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / 1000);

    // Rescale the prediction back to original image
    adjust_boxes(prediction_nms_list, image_width, image_height, input_width, input_height);

    // Show detection result
    MNN_PRINT("Detection result:\n");
    for(auto prediction_nms : prediction_nms_list) {
        MNN_PRINT("%s %f (%d, %d) (%d, %d)\n", classes[prediction_nms.class_index].c_str(), prediction_nms.confidence, int(prediction_nms.x), int(prediction_nms.y), int(prediction_nms.x + prediction_nms.width), int(prediction_nms.y + prediction_nms.height));
    }

    return;
}


int main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"mnn_model", required_argument, nullptr, 'm'},
        {"image", required_argument, nullptr, 'i'},
        {"classes", required_argument, nullptr, 'l'},
        {"anchors", required_argument, nullptr, 'a'},
        {"input_mean", required_argument, nullptr, 'b'},
        {"input_std", required_argument, nullptr, 's'},
        {"threads", required_argument, nullptr, 't'},
        {"count", required_argument, nullptr, 'c'},
        {"warmup_runs", required_argument, nullptr, 'w'},
        //{"verbose", required_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv,
                    "a:b:c:hi:l:m:s:t:w:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.anchors_file_name = optarg;
        break;
      case 'b':
        s.input_mean = strtod(optarg, nullptr);
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_img_name = optarg;
        break;
      case 'l':
        s.classes_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 's':
        s.input_std = strtod(optarg, nullptr);
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      //case 'v':
        //s.verbose =
            //strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        //break;
      case 'w':
        s.number_of_warmup_runs =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'h':
      case '?':
      default:
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
    }
  }
  RunInference(&s);
  return 0;
}

