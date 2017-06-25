//HSVによる2値化
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
using namespace std;

cv::Mat_<bool> binarize(cv::Mat hsv, int hmin, int hmax, int smin, int smax, int vmin, int vmax) {
	int h = hsv.rows;
	int w = hsv.cols;
	cv::Mat_<unsigned char> binary(h, w);

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			cv::Vec3b color = hsv.at<cv::Vec3b>(y, x);
			int h = color[0];
			int s = color[1];
			int v = color[2];
			binary[y][x] = (hmin <= h && h <= hmax && smin <= s && s <= smax && vmin <= v && v <= vmax);
		}
	}
	return binary;
}

//maskがtrueになっている画素だけ, dstにsrcをはり付ける
void paste(cv::Mat src, cv::Mat_<bool> mask, cv::Mat dst) {
	for (int y = 0; y < src.rows; y++) {
		for (int x = 0; x < src.cols; x++) {
			if (mask(y, x)) {
				dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(y, x);
			}
		}
	}
}

cv::Mat margeFigure(cv::Mat frame, cv::Mat kiseki) {
	int h = frame.rows;
	int w = frame.cols;
	cv::Mat ret = cv::Mat(cv::Size(w, h), CV_8UC3);

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			cv::Vec3b color = kiseki.at<cv::Vec3b>(y, x);
			if (color == cv::Vec3b(0, 0, 0)) {
				ret.at<cv::Vec3b>(y, x) = frame.at<cv::Vec3b>(y, x);
			}
			else {
				ret.at<cv::Vec3b>(y, x) = color;
			}
		}
	}
	return ret;
}

//副作用
cv::Mat reverseX(cv::Mat frame) {
	int l = 0;
	int r = frame.cols - 1;
	while (l < r) {
		for (int y = 0; y < frame.rows; y++) {
			std::swap(frame.at<cv::Vec3b>(y, l), frame.at<cv::Vec3b>(y, r));
		}
		l++;
		r--;
	}
	return frame;
}

cv::Mat coverMask(cv::Mat frame, cv::Mat_<bool> mask, cv::Vec3b defaultBGR = cv::Vec3b(0, 0, 0)) {
	int h = frame.rows;
	int w = frame.cols;
	cv::Mat result = cv::Mat(cv::Size(w, h), CV_8UC3);

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			if (mask[y][x]) result.at<cv::Vec3b>(y, x) = frame.at<cv::Vec3b>(y, x);
			else result.at<cv::Vec3b>(y, x) = defaultBGR;
		}
	}
	return result;
}

int main()
{
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) return -1;

	cv::namedWindow("HSVで2値化", CV_WINDOW_NORMAL);
	cv::namedWindow("HSVの調整", CV_WINDOW_NORMAL);

	int hmin = 0, hmax = 10, smin = 200, smax = 255, vmin = 80, vmax = 255;
	cv::createTrackbar("Hmin", "HSVの調整", &hmin, 180);
	cv::createTrackbar("Hmax", "HSVの調整", &hmax, 180);
	cv::createTrackbar("Smin", "HSVの調整", &smin, 256);
	cv::createTrackbar("Smax", "HSVの調整", &smax, 256);
	cv::createTrackbar("Vmin", "HSVの調整", &vmin, 256);
	cv::createTrackbar("Vmax", "HSVの調整", &vmax, 256);

	cv::Mat kiseki;

	char bkey = ' ';
	while (true) {
		cv::Mat frame;
		cap >> frame;

		//HSVによる2値化(明るい部分だけを抽出)
		cv::Mat hsv;
		cv::cvtColor(frame, hsv, CV_BGR2HSV);
		cv::Mat_<bool> mask = binarize(hsv, hmin, hmax, smin, smax, vmin, vmax);

		//キーボード操作
		char key = cv::waitKey(1);
		if (key == 'q') break;
		if (kiseki.empty() || (bkey != 'i' && key == 'i')) {
			kiseki = cv::Mat(cv::Size(frame.cols, frame.rows), CV_8UC3, cv::Scalar(0, 0, 0));
		}
		bkey = key;

		//軌跡の蓄積
		paste(frame, mask, kiseki);

		//表示
		cv::Mat result = margeFigure(frame, kiseki);
		reverseX(result);
		cv::imshow("HSVで2値化", result);
		cv::imshow("HSVの調整", reverseX(coverMask(frame, mask)));
	}
	cv::destroyAllWindows();
	return 0;
}