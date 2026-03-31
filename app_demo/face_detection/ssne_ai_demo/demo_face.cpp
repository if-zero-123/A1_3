/*
 * @Filename: demo_face.cpp
 * @Author: Hongying He
 * @Email: hongying.he@smartsenstech.com
 * @Date: 2025-12-30 14-57-47
 * @Copyright (c) 2025 SmartSens
 */
#include <fstream>
#include <iostream>
#include <cstring>
#include <fcntl.h>
#include <regex>
#include <dirent.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include "include/utils.hpp"

using namespace std;

// 全局变量和同步对象
std::mutex mtx_image;                           // 图像队列互斥锁
std::condition_variable cv_image_ready;          // 图像就绪条件变量
std::atomic<bool> stop_inference(false);         // 停止推理标志
std::atomic<int> frame_count(0);                 // 帧计数器
VISUALIZER* g_visualizer = nullptr;              // 全局可视化器指针


// 图像队列结构
struct ImagePair {
    ssne_tensor_t img1;
    ssne_tensor_t img2;
    int frame_id;
};

std::queue<ImagePair> image_queue;               // 图像队列
const int MAX_QUEUE_SIZE = 2;                    // 最大队列长度

// 全局退出标志（线程安全）
bool g_exit_flag = false;
// 保护退出标志的互斥锁
std::mutex g_mtx;

/**
 * @brief 键盘监听程序，用于结束demo
 */
void keyboard_listener() {
    std::string input;
    std::cout << "键盘监听线程已启动，输入 'q' 退出程序..." << std::endl;

    while (true) {
        // 读取键盘输入（会阻塞直到有输入）
        std::cin >> input;

        // 加锁修改退出标志
        std::lock_guard<std::mutex> lock(g_mtx);
        if (input == "q" || input == "Q") {
            g_exit_flag = true;
            std::cout << "检测到退出指令，通知主线程退出..." << std::endl;
            break;
        } else {
            std::cout << "输入无效（仅 'q' 有效），请重新输入：" << std::endl;
        }
    }
}

/**
 * @brief 检查退出标志的辅助函数（线程安全）
 * @return 是否需要退出
 */
bool check_exit_flag() {
    std::lock_guard<std::mutex> lock(g_mtx);
    return g_exit_flag;
}


/**
 * @brief 推理线程函数：从队列中获取图像并执行人脸检测
 */
void inference_thread_func(SCRFDGRAY* detector, int dual_display_offset_y) {
    cout << "[Thread] Inference thread started!" << endl;
    
    // 人脸检测结果初始化
    FaceDetectionResult* det_result1 = new FaceDetectionResult;
    // FaceDetectionResult* det_result2 = new FaceDetectionResult;
    
    while (!stop_inference) {
        ImagePair img_pair;
        bool has_image = false;
        
        // 从队列中获取图像
        {
            std::unique_lock<std::mutex> lock(mtx_image);
            
            // 等待图像就绪或停止信号
            cv_image_ready.wait(lock, [] {
                return !image_queue.empty() || stop_inference;
            });
            
            if (stop_inference && image_queue.empty()) {
                break;
            }
            
            if (!image_queue.empty()) {
                img_pair = image_queue.front();
                image_queue.pop();
                has_image = true;
            }
        }
        
        if (!has_image) {
            continue;
        }
        
        // 执行人脸检测（非阻塞主循环）
        detector->Predict(&img_pair.img1, det_result1, 0.4f);
        // detector->Predict(&img_pair.img2, det_result2, 0.4f);
        
        // 处理检测结果 - 右侧图像
        if (det_result1->boxes.size() > 0) {
            std::vector<std::array<float, 4>> boxes_original_coord;  // 存储转换后的原图坐标
            for (size_t i = 0; i < det_result1->boxes.size(); i++) {
                float x1_orig = det_result1->boxes[i][0];
                float y1_orig = det_result1->boxes[i][1];
                float x2_orig = det_result1->boxes[i][2];
                float y2_orig = det_result1->boxes[i][3];
                
                printf("[Frame %d] Right Face detected: (%.2f, %.2f, %.2f, %.2f)\n", 
                       img_pair.frame_id, x1_orig, y1_orig, x2_orig, y2_orig);
                // 保存原图坐标用于OSD绘制
                boxes_original_coord.push_back({x1_orig, y1_orig, x2_orig, y2_orig});
            }
            // 绘制检测框到OSD
            if (g_visualizer != nullptr) {
                printf("osd draw\n");
                g_visualizer->Draw(boxes_original_coord);
            }
        } else {
            // 未检测到人脸，清除OSD上的检测框
            if (g_visualizer != nullptr) {
                std::vector<std::array<float, 4>> empty_boxes;
                g_visualizer->Draw(empty_boxes);  // 传入空向量清除显示
            }
        }
        
        // // 处理检测结果 - 左侧图像
        // if (det_result2->boxes.size() > 0) {
        //     for (size_t i = 0; i < det_result2->boxes.size(); i++) {
        //         float x3_orig = det_result2->boxes[i][0];
        //         float y3_orig = det_result2->boxes[i][1] + dual_display_offset_y;
        //         float x4_orig = det_result2->boxes[i][2];
        //         float y4_orig = det_result2->boxes[i][3] + dual_display_offset_y;
                
        //         printf("[Frame %d] Left Face detected: (%.2f, %.2f, %.2f, %.2f)\n", 
        //                img_pair.frame_id, x3_orig, y3_orig, x4_orig, y4_orig);
        //     }
        // } else {
        //     cout << "[Frame " << img_pair.frame_id << "] Left No face detected" << endl;
        // }
    }
    
    // 释放资源
    delete det_result1;
    // delete det_result2;
    
    cout << "[Thread] Inference thread stopped!" << endl;
}

/**
 * @brief 人脸检测演示程序主函数
 * @return 执行结果，0表示成功
 */
int main() {
    /******************************************************************************************
     * 1. 参数配置
     ******************************************************************************************/
    
    uint8_t load_flag = 0;  // 0: 当前load偶帧; 1: 当前load奇帧（初始值为0）
    // 图像尺寸配置（根据镜头参数修改）
    int img_width = 640;    // 输入图像宽度
    int img_height = 480;  // 输入图像高度
    
    // 模型配置参数
    array<int, 2> det_shape = {640, 480};  // 检测模型输入尺寸
    string path_det = "/app_demo/app_assets/models/face_640x480.m1model";  // 人脸检测模型路径
    
    /******************************************************************************************
     * 2. 系统初始化
     ******************************************************************************************/
    
    // SSNE初始化
    if (ssne_initial()) {
        fprintf(stderr, "SSNE initialization failed!\n");
    }
    
    // 图像处理器初始化
    array<int, 2> img_shape = {img_width, img_height};  // 原始图像尺寸
    const int dual_display_offset_y = 480;  // 双目显示时第二路图像Y方向的偏移量（上下拼接显示）
    // 原图: 640×480, 模型输入图：640×480
    // 输入比例一致，无需额外裁剪偏移量
    
    // OSD可视化器初始化（用于绘制检测框）
    VISUALIZER visualizer;
    visualizer.Initialize(img_shape);  // 初始化可视化器（配置图像尺寸）
    
    // 设置全局可视化器指针供推理线程使用
    g_visualizer = &visualizer;

    IMAGEPROCESSOR processor;
    processor.Initialize(&img_shape);  // 初始化图像处理器（配置原图尺寸）
    
    // 人脸检测模型初始化
    SCRFDGRAY detector;
    int box_len = det_shape[0] * det_shape[1] / 512 * 21;  // 计算最大检测框数量
    detector.Initialize(path_det, &img_shape, &det_shape, false, box_len);  // 初始化检测器
    
    cout << "[INFO] Detection Model initialized!" << endl;    
    // 系统稳定等待
    cout << "sleep for 0.2 second!" << endl;
    sleep(0.2);  // 等待系统稳定
    
    // 启动推理线程
    std::thread inference_thread(inference_thread_func, &detector, dual_display_offset_y);
    cout << "[INFO] Inference thread started!" << endl;

    // 创建键盘监听线程
    std::thread listener_thread(keyboard_listener);
    
    uint16_t num_frames = 0;  // 帧计数器
    ssne_tensor_t img_sensor[2];  // 图像tensor定义
    ssne_tensor_t output_sensor[2];
    output_sensor[0] = create_tensor(img_width, img_height * 2, SSNE_Y_8, SSNE_BUF_AI);
    output_sensor[1] = create_tensor(img_width, img_height * 2, SSNE_Y_8, SSNE_BUF_AI);
    
    processor.GetDualImage(&img_sensor[0], &img_sensor[1]);
	copy_double_tensor_buffer(img_sensor[0], img_sensor[1], output_sensor[0]);
	copy_double_tensor_buffer(img_sensor[0], img_sensor[1], output_sensor[1]);
 
    int res = set_isp_debug_config(output_sensor[0], output_sensor[1]);

    while (num_frames < 2) {
		res = start_isp_debug_load();
		num_frames++;
	}
    /******************************************************************************************
     * 3. 主处理循环 - 只负责图像拷贝到ISP debug，不阻塞推理
     ******************************************************************************************/
    //循环5000帧后推出，循环次数可以修改，也可以改成while(true)
    while (!check_exit_flag()) {
        
        // 从sensor获取图像（裁剪图）
        processor.GetDualImage(&img_sensor[0], &img_sensor[1]);

        get_even_or_odd_flag(load_flag);

        if (load_flag == 0)
        {
            copy_double_tensor_buffer(img_sensor[0], img_sensor[1], output_sensor[0]);
        }
        else
        {
            copy_double_tensor_buffer(img_sensor[0], img_sensor[1], output_sensor[1]);
        }

        // 启动ISP debug load（主循环核心任务，不能被阻塞）
        res = start_isp_debug_load();
        
        // 将图像放入队列供推理线程使用（非阻塞）
        {
            std::unique_lock<std::mutex> lock(mtx_image);
            
            // 如果队列未满，则添加图像
            if (image_queue.size() < MAX_QUEUE_SIZE) {
                ImagePair img_pair;
                img_pair.img1 = img_sensor[0];
                img_pair.img2 = img_sensor[1];
                img_pair.frame_id = num_frames;
                
                image_queue.push(img_pair);
                cv_image_ready.notify_one();  // 通知推理线程
            }
            // 如果队列满了，跳过本帧推理（避免阻塞）
        }

        num_frames += 1;  // 帧计数器递增
    }
    
    /******************************************************************************************
     * 4. 停止推理线程并等待其完成
     ******************************************************************************************/
    
    cout << "[INFO] Main loop finished, stopping inference thread..." << endl;

    // 等待监听线程退出，释放资源
    if (listener_thread.joinable()) {
        listener_thread.join();
    }
    
    // 设置停止标志并通知推理线程
    {
        std::unique_lock<std::mutex> lock(mtx_image);
        stop_inference = true;
        cv_image_ready.notify_one();
    }
    
    // 等待推理线程退出
    if (inference_thread.joinable()) {
        inference_thread.join();
        cout << "[INFO] Inference thread joined successfully!" << endl;
    }
    
    /******************************************************************************************
     * 5. 资源释放
     ******************************************************************************************/
    
    detector.Release();  // 释放检测器资源
    processor.Release();  // 释放图像处理器资源
    visualizer.Release();  // 释放可视化器资源
    
    if (ssne_release()) {
        fprintf(stderr, "SSNE release failed!\n");
        return -1;
    }
    
    return 0;
}
 
