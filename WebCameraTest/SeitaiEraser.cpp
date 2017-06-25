//�E3�t���[���ԍ��� (�F���ς������f�͂��̂܂�, �F���ς��Ȃ���f�͍����\���j
//�E���x�����O�����i�m�C�Y����, �������𖄂߂鏈��)
//�E�f�B���C�o�[(�t���[�����Ŏw��)
//�E�V���b�^�[�@�\ (�L�[�{�[�h��s�{�^���������Ă���, shutter_frame�t���[����Ɏʐ^���B��)
//���������܂��i���[�������߂�j�F
//�P�D�}�X�N(2�l)�摜�ɂ͔ԕ���u���B�J���[�摜�E�Z�W�摜�ɂ͔ԕ���u���Ȃ��B
//�Q�D�֐��͊�{�I�ɕ���p�����i�l��Ԃ��̂ł͂Ȃ��A���ڒl��ύX���銴���j
//�R. ����ł��x���ꍇ�́A�C���e�O�����݌v������B
//�S�D��f��(�c, ��)�͒萔�Ƃ��܂��B�摜�Ԃ̂����ɂ��āA�������m�ۂ��f�����킹�͂���Ă�����̂Ƃ��܂��B

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <queue>
#include <algorithm>
using namespace std;

int Abs(int x) { if (x > 0) return x; return -x; }

//���E���]
void reverseX(cv::Mat frame) {
	int l = 0;
	int r = frame.cols - 1;
	while (l < r) {
		for (int y = 0; y < frame.rows; y++) {
			std::swap(frame.at<cv::Vec3b>(y, l), frame.at<cv::Vec3b>(y, r));
		}
		l++;
		r--;
	}
}

//��ʂ�, �q���Ɋ�����F�ق�, �E�F�C�g�����������ق����悢�B
void cvtColor(cv::Mat frame, cv::Mat_<unsigned char> gray, double weightR, double weightG, double weightB){

	if (weightR + weightG + weightB == 0) { weightR = 0.299; weightG = 0.587; weightB = 0.114; }
	double scale = weightR + weightG + weightB;
	weightR /= scale;
	weightG /= scale;
	weightB /= scale;

	for (int y = 0; y < frame.rows; y++) {
		for (int x = 0; x < frame.cols; x++) {
			cv::Vec3b color = frame.at<cv::Vec3b>(y, x);
			int newColor = weightR * color[2] + weightG * color[1] + weightB * color[0];
			newColor = min(255, max(0, newColor));

			//��ʂ�, �����ق�newColor�̒l������ (weight����0�̂Ƃ�)
			gray(y, x) = newColor;
		}
	}
}

//���̌��o (���̕����Ȃ�true(��), ����ȊO�Ȃ�false(��), �ɂ���2�l�摜) -> dmask�̊O���͔ԕ�(false)�ň͂��Ă���Ƃ���
void doutaiMask(cv::Mat_<unsigned char> gray[3], cv::Mat_<bool> dmask, int sikii = 1) {
	int h = gray[0].rows;
	int w = gray[0].cols;

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			dmask(y + 1, x + 1) = (Abs(gray[0](y, x) - (int)gray[1](y, x)) >= sikii && Abs(gray[1](y, x) - (int)gray[2](y, x)) >= sikii);
		}
	}
}

//���׃����O(0�Ԃ���) + �ʐ�K�ȉ��̍폜 + �w�i�ȊO�̍������𔒂�����
void labelingKakou(cv::Mat_<bool> dmask, int lowArea) {
	const int dy[4] = { -1, 0, 1, 0 };
	const int dx[4] = { 0, 1, 0, -1 };
	int h = dmask.rows;
	int w = dmask.cols;
	int initValue = -1;
	cv::Mat_<int> label(h, w, initValue);
	int labelNum = 0;

	vector<bool> isToBlack;
	queue<int> que;

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			if (label[y][x] != -1) continue;
			int areaSize = 0;
			que.push(y);
			que.push(x);
			label[y][x] = labelNum; areaSize++;
			while (!que.empty()) {
				int sy = que.front(); que.pop();
				int sx = que.front(); que.pop();
				for (int dir = 0; dir < 4; dir++) {
					int ny = sy + dy[dir];
					int nx = sx + dx[dir];
					if (ny < 0 || ny >= h || nx < 0 || nx >= w || dmask[sy][sx] != dmask[ny][nx] || label[ny][nx] != -1) continue;
					label[ny][nx] = labelNum; areaSize++;
					que.push(ny);
					que.push(nx);
				}
			}
			if ((!dmask[y][x] && labelNum == 0) || (dmask[y][x] && areaSize <= lowArea)) isToBlack.push_back(true);
			else isToBlack.push_back(false);
			labelNum++;
		}
	}

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			dmask[y][x] = !isToBlack[label[y][x]];
		}
	}
}


//�\���摜�̐��� (���̂̓J���[��, ����ȊO�̓f�t�H���g�F��)
void eraseDoutai(cv::Mat frame, cv::Mat_<unsigned char> doutaiMask, cv::Mat result, int defaultR = 0, int defaultG = 0, int defaultB = 0) {
	int h = frame.rows;
	int w = frame.cols;

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			if (!doutaiMask(y + 1, x + 1)) {	//�����Ă��Ȃ�
				result.at<cv::Vec3b>(y, x) = cv::Vec3b(defaultB, defaultG, defaultR);
			}
			else {	//�����Ă���
				result.at<cv::Vec3b>(y, x) = frame.at<cv::Vec3b>(y, x);
			}
		}
	}
}

//�{��
cv::Mat frames[3];					//frames[0]����ԐV����
cv::Mat_<unsigned char> grays[3];
cv::Mat_<bool> dmask;
cv::Mat result;
queue<cv::Mat> que;

int main()
{
	cv::VideoCapture cap(1);
	if (!cap.isOpened()) return -1;

	int lowArea = 10;
	int delayFrame = 0;
	int shutterFrame = 50;
	cv::namedWindow("SeitaiEraser", CV_WINDOW_NORMAL);
	cv::namedWindow("Sonomama", CV_WINDOW_NORMAL);
	cv::createTrackbar("lowArea", "SeitaiEraser", &lowArea, 50);
	cv::createTrackbar("delay", "SeitaiEraser", &delayFrame, 100);
	cv::createTrackbar("shutterFrame", "SeitaiEraser", &shutterFrame, 100);

	int INF = 11451419;
	int shutterRemFrame = INF;	//����shutterRemFrame�t���[����ɕ\�����Ă���摜���B�e����

	int loopCnt = 0;
	char bkey = ' ';
	while (true)
	{
		cv::Mat frame;
		cap >> frame;
		reverseX(frame);
		while (que.size() > delayFrame) que.pop();
		que.push(frame);

		//������
		if (loopCnt == 0) {
			unsigned char uc_zero = 0;
			bool b_zero = false;
			dmask = cv::Mat_<bool>(frame.rows + 2, frame.cols + 2, b_zero);
			result = cv::Mat(cv::Size(frame.cols, frame.rows), CV_8UC3, cv::Scalar(0, 0, 0));
		}

		//delayFrame�O�̉摜
		cv::Mat seeFrame = que.front();

		//�ۑ�
		frames[2] = frames[1];
		frames[1] = frames[0];
		frames[0] = seeFrame;
		grays[2] = grays[1];
		grays[1] = grays[0];
		grays[0] = cv::Mat_<unsigned char>(frame.rows, frame.cols);
		cvtColor(seeFrame, grays[0], 0, 0, 0);

		//�G���[�΍�
		loopCnt++;
		if (loopCnt < 3) continue;

		//���̌��o (1�s�ځc�}�X�N����, 2�`9�s�ځc�m�C�Y�����Ȃ�)
		doutaiMask(grays, dmask, 5);	//doutaiMask�̑�2������傫������ق�, �u���̂Ɣ��肳���̈�v���������Ȃ�܂�
		labelingKakou(dmask, lowArea);	//�m�C�Y����

		//���̂�����\�������摜
		eraseDoutai(frames[1], dmask, result);

		//��~����, �B�e, �J�E���g�\��
		char key = cv::waitKey(1);
		if (key == 'q') break;

		if (bkey != 's' && key == 's') {
			shutterRemFrame = shutterFrame;
		}
		if (shutterRemFrame == 0) {
			shutterRemFrame = INF;
			cv::imwrite("shot.png", result);
			cv::imwrite("shot_sonomama.png", seeFrame);

			//�m�C�Y�����O�̉摜
			cv::Mat_<bool> mask = dmask.clone();
			doutaiMask(grays, mask, 5);
			cv::Mat tyuukan = result.clone();
			eraseDoutai(frames[1], mask, tyuukan);
			cv::imwrite("shot_noizExist.png", tyuukan);
		}
		if (shutterRemFrame > 0 && shutterRemFrame <= shutterFrame) {
			string str = cv::format("%d", shutterRemFrame);
			cv::putText(result, str, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2, CV_AA);
		}
		shutterRemFrame--;
		bkey = key;

		//�摜�\��
		imshow("SeitaiEraser", result);
		imshow("Sonomama", seeFrame);
	}
	cv::destroyAllWindows();
	return 0;
}