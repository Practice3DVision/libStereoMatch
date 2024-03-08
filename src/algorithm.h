/**
 * @file algorithm.h
 * @author Liu Yunhuang (1369215984@qq.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */

 #ifndef __ALGORITHM_H_
 #define __ALGORITHM_H_

 #include <typeDef.h>

 namespace cv {
    class Mat;
 }

 namespace libSM {
    class Algorithm {
        public:
            virtual ~Algorithm() {}
            /**
             * @brief perform stereo matching
             * 
             * @param left left image
             * @param right right image
             * @param dispMap disparity Map
             */
            virtual void match(IN const cv::Mat& left, IN const cv::Mat& right, OUT cv::Mat& dispMap) = 0;
    };
 }

 #endif //!__ALGORITHM_H_