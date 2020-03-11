// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <k4a/k4a.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

static void create_xy_table(const k4a_calibration_t* calibration, k4a_image_t xy_table)
{
    k4a_float2_t* table_data = (k4a_float2_t*)(void*)k4a_image_get_buffer(xy_table);

    int width = calibration->depth_camera_calibration.resolution_width;
    int height = calibration->depth_camera_calibration.resolution_height;

    k4a_float2_t p;
    k4a_float3_t ray;
    int valid;

    for (int y = 0, idx = 0; y < height; y++)
    {
        p.xy.y = (float)y;
        for (int x = 0; x < width; x++, idx++)
        {
            p.xy.x = (float)x;

            k4a_calibration_2d_to_3d(
                calibration, &p, 1.f, K4A_CALIBRATION_TYPE_DEPTH, K4A_CALIBRATION_TYPE_DEPTH, &ray, &valid);

            if (valid)
            {
                table_data[idx].xy.x = ray.xyz.x;
                table_data[idx].xy.y = ray.xyz.y;
            }
            else
            {
                table_data[idx].xy.x = nanf("");
                table_data[idx].xy.y = nanf("");
            }
        }
    }
}

static void generate_point_cloud(const k4a_image_t depth_image,
    const k4a_image_t xy_table,
    k4a_image_t point_cloud,
    int* point_count)
{
    int width = k4a_image_get_width_pixels(depth_image);
    int height = k4a_image_get_height_pixels(depth_image);

    uint16_t* depth_data = (uint16_t*)(void*)k4a_image_get_buffer(depth_image);
    k4a_float2_t* xy_table_data = (k4a_float2_t*)(void*)k4a_image_get_buffer(xy_table);
    k4a_float3_t* point_cloud_data = (k4a_float3_t*)(void*)k4a_image_get_buffer(point_cloud);

    *point_count = 0;
    for (int i = 0; i < width * height; i++)
    {
        if (depth_data[i] != 0 && !isnan(xy_table_data[i].xy.x) && !isnan(xy_table_data[i].xy.y))
        {
            point_cloud_data[i].xyz.x = xy_table_data[i].xy.x * (float)depth_data[i];
            point_cloud_data[i].xyz.y = xy_table_data[i].xy.y * (float)depth_data[i];
            point_cloud_data[i].xyz.z = (float)depth_data[i];
            (*point_count)++;
        }
        else
        {
            point_cloud_data[i].xyz.x = nanf("");
            point_cloud_data[i].xyz.y = nanf("");
            point_cloud_data[i].xyz.z = nanf("");
        }
    }
}

static void write_point_cloud(const char* file_name, const k4a_image_t Master_point_cloud, const k4a_image_t Sub_point_cloud, int point_count)
{
    int Master_width = k4a_image_get_width_pixels(Master_point_cloud);
    int Master_height = k4a_image_get_height_pixels(Master_point_cloud);
    k4a_float3_t* Master_point_cloud_data = (k4a_float3_t*)(void*)k4a_image_get_buffer(Master_point_cloud);

    int Sub_width = k4a_image_get_width_pixels(Sub_point_cloud);
    int Sub_height = k4a_image_get_height_pixels(Sub_point_cloud);
    k4a_float3_t* Sub_point_cloud_data = (k4a_float3_t*)(void*)k4a_image_get_buffer(Sub_point_cloud);

    // save to the ply file
    std::ofstream ofs(file_name); // text mode first
    ofs << "ply" << std::endl;
    ofs << "format ascii 1.0" << std::endl;
    ofs << "element vertex"
        << " " << point_count << std::endl;
    ofs << "property float x" << std::endl;
    ofs << "property float y" << std::endl;
    ofs << "property float z" << std::endl;
    ofs << "end_header" << std::endl;
    ofs.close();

    std::stringstream ss;
    for (int i = 0; i < Master_width * Master_height; i++)
    {
        if (isnan(Master_point_cloud_data[i].xyz.x) || isnan(Master_point_cloud_data[i].xyz.y) || isnan(Master_point_cloud_data[i].xyz.z))
        {
            continue;
        }

        ss << (float)Master_point_cloud_data[i].xyz.x << " " << (float)Master_point_cloud_data[i].xyz.y << " "
            << (float)Master_point_cloud_data[i].xyz.z << std::endl;
    }
    for (int i = 0; i < Sub_width * Sub_height; i++)
    {
        if (isnan(Sub_point_cloud_data[i].xyz.x) || isnan(Sub_point_cloud_data[i].xyz.y) || isnan(Sub_point_cloud_data[i].xyz.z))
        {
            continue;
        }

        ss << (float)Sub_point_cloud_data[i].xyz.x << " " << (float)Sub_point_cloud_data[i].xyz.y << " "
            << (float)Sub_point_cloud_data[i].xyz.z << std::endl;
    }

    std::ofstream ofs_text(file_name, std::ios::out | std::ios::app);
    ofs_text.write(ss.str().c_str(), (std::streamsize)ss.str().length());
}

void calibrateSub2Master(const k4a_image_t depth_image, cv::Vec3d masterRvec, cv::Vec3d masterTvec, cv::Vec3d subRvec, cv::Vec3d subTvec, const k4a_image_t Subpoint_cloud, k4a_image_t Subpoint_cloud_InMaster)
{
    Mat masterRvecTrans, subRvecTrans;//将罗德里格斯转化成矩阵形式后的，均为从marker到相机的
    Rodrigues(masterRvec, masterRvecTrans);
    Rodrigues(subRvec, subRvecTrans);

    Mat Rsub2master, subRvecTrans;//
    Rsub2master=masterRvecTrans.inv();

    int width = k4a_image_get_width_pixels(depth_image);
    int height = k4a_image_get_height_pixels(depth_image);
    k4a_float3_t* Subpoint_cloud_data = (k4a_float3_t*)(void*)k4a_image_get_buffer(Subpoint_cloud);
    k4a_float3_t* Subpoint_cloud_data_InMaster = (k4a_float3_t*)(void*)k4a_image_get_buffer(Subpoint_cloud_InMaster);


    for (int i = 0; i < width * height; i++)
    {
        if(Subpoint_cloud_data[i].xyz.z != 0 && !isnan(Subpoint_cloud_data[i].xyz.x) && !isnan(Subpoint_cloud_data[i].xyz.y))//根据rvec和tvec将其转入到主设备坐标系下
        {
            //从sub相机坐标系转到marker坐标系
            float tempX = subTvec[0] + Rsub2master.ptr<float>(0)[0] * Subpoint_cloud_data[i].xyz.x + Rsub2master.ptr<float>(0)[1] * Subpoint_cloud_data[i].xyz.y + Rsub2master.ptr<float>(0)[2] * Subpoint_cloud_data[i].xyz.z;
            float tempY = subTvec[1] + Rsub2master.ptr<float>(1)[0] * Subpoint_cloud_data[i].xyz.x + Rsub2master.ptr<float>(1)[1] * Subpoint_cloud_data[i].xyz.y + Rsub2master.ptr<float>(1)[2] * Subpoint_cloud_data[i].xyz.z;
            float tempZ = subTvec[2] + Rsub2master.ptr<float>(2)[0] * Subpoint_cloud_data[i].xyz.x + Rsub2master.ptr<float>(2)[1] * Subpoint_cloud_data[i].xyz.y + Rsub2master.ptr<float>(2)[2] * Subpoint_cloud_data[i].xyz.z;

            //从marker坐标系转到master坐标系
            Subpoint_cloud_data_InMaster[i].xyz.x = masterTvec[0] + masterRvecTrans.ptr<float>(0)[0] * tempX + masterRvecTrans.ptr<float>(0)[1] * tempY + masterRvecTrans.ptr<float>(0)[2] * tempZ;
            Subpoint_cloud_data_InMaster[i].xyz.y = masterTvec[1] + masterRvecTrans.ptr<float>(1)[0] * tempX + masterRvecTrans.ptr<float>(1)[1] * tempY + masterRvecTrans.ptr<float>(1)[2] * tempZ;
            Subpoint_cloud_data_InMaster[i].xyz.z = masterTvec[2] + masterRvecTrans.ptr<float>(2)[0] * tempX + masterRvecTrans.ptr<float>(2)[1] * tempY + masterRvecTrans.ptr<float>(2)[2] * tempZ;

        }
        else
        {
            Subpoint_cloud_data_InMaster[i].xyz.x = nanf("");
            Subpoint_cloud_data_InMaster[i].xyz.y = nanf("");
            Subpoint_cloud_data_InMaster[i].xyz.z = nanf("");
        }
    }

}


void drawCalibrateMarker()//用来画标定用的marker
{
    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(5, 7, 0.04, 0.01, dictionary);
    cv::Mat boardImage;
    board->draw(cv::Size(600, 500), boardImage, 10, 1);
    imwrite("MarkerImage.jpg", boardImage);
}
int main(void)
{
    int returnCode = 1;
    
    const int32_t TIMEOUT_IN_MS = 1000;


    
    std::string file_name;
    uint32_t device_count = 0;
    
    

    //Mater Kinect
    k4a_device_t Mater_device = NULL;
    k4a_capture_t Mater_capture = NULL;
    k4a_image_t Mater_depth_image = NULL;
    k4a_image_t Mater_color_image = NULL;
    k4a_image_t Mater_xy_table = NULL;
    k4a_image_t Mater_point_cloud = NULL;
    k4a_device_configuration_t Mater_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    Mater_config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
    Mater_config.camera_fps = K4A_FRAMES_PER_SECOND_30;
    k4a_calibration_t Mater_calibration;
    Mat Master_camera_matrix_color = Mat::zeros(3, 3, CV_32F);
    Mat Master_dist_coeffs_color = Mat::zeros(5, 1, CV_32F);
    int Mater_point_count = 0;




    //Sub Kinect
    k4a_device_t Sub_device = NULL;
    k4a_capture_t Sub_capture = NULL;
    k4a_image_t Sub_depth_image = NULL;
    k4a_image_t Sub_color_image = NULL;
    k4a_image_t Sub_xy_table = NULL;
    k4a_image_t Sub_point_cloud = NULL;
    k4a_image_t Sub_point_cloudInMaster = NULL;
    k4a_device_configuration_t Sub_config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    Sub_config.depth_mode = K4A_DEPTH_MODE_WFOV_2X2BINNED;
    Sub_config.camera_fps = K4A_FRAMES_PER_SECOND_30;
    k4a_calibration_t Sub_calibration;
    Mat Sub_camera_matrix_color = Mat::zeros(3, 3, CV_32F);
    Mat Sub_dist_coeffs_color = Mat::zeros(5, 1, CV_32F);
    int Sub_point_count = 0;



    device_count = k4a_device_get_installed_count();

    if (device_count == 0)
    {
        printf("No K4A devices found\n");
        return 0;
    }
    if (device_count == 1)
    {
        printf("Only a K4A devices found\n");
        return 0;
    }

    if (K4A_RESULT_SUCCEEDED != k4a_device_open(K4A_DEVICE_DEFAULT, &Sub_device))
    {
        printf("Failed to open sub device\n");
        goto Exit;
    }
    if (K4A_RESULT_SUCCEEDED != k4a_device_open(K4A_DEVICE_DEFAULT, &Mater_device))
    {
        printf("Failed to open master device\n");
        goto Exit;
    }
    if (K4A_RESULT_SUCCEEDED != k4a_device_get_calibration(Mater_device, Mater_config.depth_mode, Mater_config.color_resolution, &Mater_calibration)
        && K4A_RESULT_SUCCEEDED != k4a_device_get_calibration(Sub_device, Sub_config.depth_mode, Sub_config.color_resolution, &Sub_calibration))
    {
        printf("Failed to get calibration\n");
        goto Exit;
    }
    k4a_calibration_intrinsic_parameters_t* Mater_intrinsics_color = &Mater_calibration.color_camera_calibration.intrinsics.parameters;
    Master_camera_matrix_color = ((cv::Mat_<float>(3, 3)) 
        << Mater_intrinsics_color->param.fx,                          0.f, Mater_intrinsics_color->param.cx,
                                  0.f, Mater_intrinsics_color->param.fy, Mater_intrinsics_color->param.cy,
                                  0.f,                          0.f,                         1.f);
    Master_dist_coeffs_color = ((cv::Mat_<float>(5, 1))
        <<  Mater_intrinsics_color->param.k1, 
            Mater_intrinsics_color->param.k2, 
            Mater_intrinsics_color->param.p1,
            Mater_intrinsics_color->param.p2, 
            Mater_intrinsics_color->param.k3);
    k4a_calibration_intrinsic_parameters_t* Sub_intrinsics_color = &Sub_calibration.color_camera_calibration.intrinsics.parameters;
    Sub_camera_matrix_color = ((cv::Mat_<float>(3, 3))
        << Sub_intrinsics_color->param.fx, 0.f, Sub_intrinsics_color->param.cx,
        0.f, Sub_intrinsics_color->param.fy, Sub_intrinsics_color->param.cy,
        0.f, 0.f, 1.f);
    Sub_dist_coeffs_color = ((cv::Mat_<float>(5, 1))
        << Sub_intrinsics_color->param.k1,
        Sub_intrinsics_color->param.k2,
        Sub_intrinsics_color->param.p1,
        Sub_intrinsics_color->param.p2,
        Sub_intrinsics_color->param.k3);



    //主设备相关
    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
        Mater_calibration.depth_camera_calibration.resolution_width,
        Mater_calibration.depth_camera_calibration.resolution_height,
        Mater_calibration.depth_camera_calibration.resolution_width * (int)sizeof(k4a_float2_t),
        &Mater_xy_table);
    create_xy_table(&Mater_calibration, Mater_xy_table);
    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
        Mater_calibration.depth_camera_calibration.resolution_width,
        Mater_calibration.depth_camera_calibration.resolution_height,
        Mater_calibration.depth_camera_calibration.resolution_width * (int)sizeof(k4a_float3_t),
        &Mater_point_cloud);

    //副设备相关
    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
        Sub_calibration.depth_camera_calibration.resolution_width,
        Sub_calibration.depth_camera_calibration.resolution_height,
        Sub_calibration.depth_camera_calibration.resolution_width * (int)sizeof(k4a_float2_t),
        &Sub_xy_table);
    create_xy_table(&Sub_calibration, Sub_xy_table);
    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
        Sub_calibration.depth_camera_calibration.resolution_width,
        Sub_calibration.depth_camera_calibration.resolution_height,
        Sub_calibration.depth_camera_calibration.resolution_width * (int)sizeof(k4a_float3_t),
        &Sub_point_cloud);
    k4a_image_create(K4A_IMAGE_FORMAT_CUSTOM,
        Sub_calibration.depth_camera_calibration.resolution_width,
        Sub_calibration.depth_camera_calibration.resolution_height,
        Sub_calibration.depth_camera_calibration.resolution_width * (int)sizeof(k4a_float3_t),
        &Sub_point_cloudInMaster);//转换坐标后在主相机中的



    if (K4A_RESULT_SUCCEEDED != k4a_device_start_cameras(Sub_device, &Sub_config))
    {
        printf("Failed to start Sub cameras\n");
        goto Exit;
    }
    if (K4A_RESULT_SUCCEEDED != k4a_device_start_cameras(Mater_device, &Mater_config))
    {
        printf("Failed to start Master cameras\n");
        goto Exit;
    }




    /*两个kinect相对位姿标定*/
    bool poseEstimationOK_master = false;
    bool poseEstimationOK_sub = false;
    cv::Vec3d Mater_rvecs, Mater_tvecs;
    cv::Vec3d Sub_rvecs, Sub_tvecs;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(5, 7, 0.04, 0.01, dictionary);
    while (1)
    {
        if (k4a_device_get_capture(Mater_device, &Mater_capture, TIMEOUT_IN_MS) && k4a_device_get_capture(Sub_device, &Sub_capture, TIMEOUT_IN_MS))
        {
            //master
            {
                Mater_color_image = k4a_capture_get_color_image(Mater_capture);
                uint16_t* colorTextureBuffer_master = (uint16_t*)(void*)k4a_image_get_buffer(Mater_color_image);
                Mat colorFrame_master = Mat(k4a_image_get_height_pixels(Mater_color_image), k4a_image_get_width_pixels(Mater_color_image), CV_16UC4, colorTextureBuffer_master);
                cvtColor(colorFrame_master, colorFrame_master, cv::COLOR_BGRA2BGR);
                {
                    //采用一个含有多个marker的板子
                    std::vector<int> ids;
                    std::vector<std::vector<cv::Point2f> > corners;
                    cv::aruco::detectMarkers(colorFrame_master, dictionary, corners, ids);
                    // if at least one marker detected
                    if (ids.size() > 0) {
                        //cv::aruco::drawDetectedMarkers(colorFrame_master, corners, ids);
                        //板子坐标系到相机坐标系的转换,采用罗德里格斯表示
                        int valid = cv::aruco::estimatePoseBoard(corners, ids, board, Master_camera_matrix_color, Master_dist_coeffs_color, Mater_rvecs, Mater_tvecs);
                        // if at least one board marker detected
                        if (valid > 0)
                            poseEstimationOK_master = true;
                        else
                            poseEstimationOK_master = false;

                    } else
                        poseEstimationOK_master = false;

                }
            }
            //sub
            {
                Sub_color_image = k4a_capture_get_color_image(Sub_capture);
                uint16_t* colorTextureBuffer_sub = (uint16_t*)(void*)k4a_image_get_buffer(Sub_color_image);
                Mat colorFrame_sub = Mat(k4a_image_get_height_pixels(Sub_color_image), k4a_image_get_width_pixels(Sub_color_image), CV_16UC4, colorTextureBuffer_sub);
                cvtColor(colorFrame_sub, colorFrame_sub, cv::COLOR_BGRA2BGR);
                {

                    //采用一个含有多个marker的板子
                    std::vector<int> ids;
                    std::vector<std::vector<cv::Point2f> > corners;
                    cv::aruco::detectMarkers(colorFrame_sub, dictionary, corners, ids);
                    // if at least one marker detected
                    if (ids.size() > 0) {
                        //cv::aruco::drawDetectedMarkers(colorFrame_master, corners, ids);
                        int valid = cv::aruco::estimatePoseBoard(corners, ids, board, Sub_camera_matrix_color, Sub_dist_coeffs_color, Sub_rvecs, Sub_tvecs);
                        // if at least one board marker detected
                        if (valid > 0)
                            poseEstimationOK_sub = true;
                        else
                            poseEstimationOK_sub = false;

                    }
                    else
                        poseEstimationOK_sub = false;
                }
            }
            if (poseEstimationOK_master && poseEstimationOK_sub)
            {
                cout << "calibrate success!" << endl;
                break;
            }
        }
    }


    /*开始采集点云*/
    //Master
    // Get a capture
    switch (k4a_device_get_capture(Mater_device, &Mater_capture, TIMEOUT_IN_MS))
    {
    case K4A_WAIT_RESULT_SUCCEEDED:
        break;
    case K4A_WAIT_RESULT_TIMEOUT:
        printf("Timed out waiting for a capture\n");
        goto Exit;
    case K4A_WAIT_RESULT_FAILED:
        printf("Failed to read a capture\n");
        goto Exit;
    }
    // Get a depth image
    Mater_depth_image = k4a_capture_get_depth_image(Mater_capture);
    if (Mater_depth_image == 0)
    {
        printf("Failed to get depth image from capture\n");
        goto Exit;
    }
    generate_point_cloud(Mater_depth_image, Mater_xy_table, Mater_point_cloud, &Mater_point_count);

    //Sub
    // Get a capture
    switch (k4a_device_get_capture(Sub_device, &Sub_capture, TIMEOUT_IN_MS))
    {
    case K4A_WAIT_RESULT_SUCCEEDED:
        break;
    case K4A_WAIT_RESULT_TIMEOUT:
        printf("Timed out waiting for a capture\n");
        goto Exit;
    case K4A_WAIT_RESULT_FAILED:
        printf("Failed to read a capture\n");
        goto Exit;
    }
    // Get a depth image
    Sub_depth_image = k4a_capture_get_depth_image(Sub_capture);
    if (Sub_depth_image == 0)
    {
        printf("Failed to get depth image from capture\n");
        goto Exit;
    }
    generate_point_cloud(Sub_depth_image, Sub_xy_table, Sub_point_cloud, &Sub_point_count);

    //vector< Mat > Mater_rvecs, Mater_tvecs;
    //vector< Mat > Sub_rvecs, Sub_tvecs;
    //把sub中的点转化到master中去;
    //calibrateSub2Master(Sub_depth_image, Sub_rvecs, Sub_tvecs, Sub_point_cloud, Sub_point_cloudInMaster)
    calibrateSub2Master(Sub_depth_image, Mater_rvecs, Mater_tvecs, Sub_rvecs, Sub_tvecs, Sub_point_cloud, Sub_point_cloudInMaster);



    write_point_cloud("point.ply", Mater_point_cloud, Sub_point_cloudInMaster, Mater_point_count+ Sub_point_count);


    k4a_image_release(Mater_depth_image);
    k4a_capture_release(Mater_capture);
    k4a_image_release(Mater_xy_table);
    k4a_image_release(Mater_point_cloud);

    k4a_image_release(Sub_depth_image);
    k4a_capture_release(Sub_capture);
    k4a_image_release(Sub_xy_table);
    k4a_image_release(Sub_point_cloud);
    k4a_image_release(Sub_point_cloudInMaster);

    returnCode = 0;
Exit:
    if (Mater_device != NULL)
    {
        k4a_device_close(Mater_device);
    }
    if (Sub_device != NULL)
    {
        k4a_device_close(Sub_device);
    }

    return returnCode;
}
