//・3フレーム間差分 (色が変わった画素はそのまま, 色が変わらない画素は黒く表示）
//・ラベリング処理（ノイズ除去, 黒い穴を埋める処理)
//・ディレイバー(フレーム数で指定)
//・シャッター機能 (キーボードのsボタンを押してから, shutter_frameフレーム後に写真を撮る)
//高速化します（ルールを決める）：
//１．マスク(2値)画像には番兵を置く。カラー画像・濃淡画像には番兵を置かない。
//２．関数は基本的に副作用を持つ（値を返すのではなく、直接値を変更する感じ）
//３. それでも遅い場合は、インテグラル設計をする。
//４．画素数(縦, 横)は定数とします。画像間のやり取りについて、メモリ確保や画素数合わせはされているものとします。

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <queue>
#include <algorithm>
using namespace std;

int Abs(int x) { if (x > 0) return x; return -x; }

//左右反転
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

//一般に, 敏感に感じる色ほど, ウェイトを高くしたほうがよい。
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

			//一般に, 白いほどnewColorの値が高い (weight○≧0のとき)
			gray(y, x) = newColor;
		}
	}
}

//動体検出 (動体部分ならtrue(白), それ以外ならfalse(黒), にした2値画像) -> dmaskの外側は番兵(false)で囲ってあるとする
void doutaiMask(cv::Mat_<unsigned char> gray[3], cv::Mat_<bool> dmask, int sikii = 1) {
	int h = gray[0].rows;
	int w = gray[0].cols;

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			dmask(y + 1, x + 1) = (Abs(gray[0](y, x) - (int)gray[1](y, x)) >= sikii && Abs(gray[1](y, x) - (int)gray[2](y, x)) >= sikii);
		}
	}
}

//ラべリング(0番から) + 面積K以下の削除 + 背景以外の黒成分を白くする
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


//表示画像の生成 (動体はカラーで, それ以外はデフォルト色で)
void eraseDoutai(cv::Mat frame, cv::Mat_<unsigned char> doutaiMask, cv::Mat result, int defaultR = 0, int defaultG = 0, int defaultB = 0) {
	int h = frame.rows;
	int w = frame.cols;

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			if (!doutaiMask(y + 1, x + 1)) {	//動いていない
				result.at<cv::Vec3b>(y, x) = cv::Vec3b(defaultB, defaultG, defaultR);
			}
			else {	//動いている
				result.at<cv::Vec3b>(y, x) = frame.at<cv::Vec3b>(y, x);
			}
		}
	}
}

//本体
cv::Mat frames[3];					//frames[0]が一番新しい
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
	int shutterRemFrame = INF;	//あとshutterRemFrameフレーム後に表示している画像を撮影する

	int loopCnt = 0;
	char bkey = ' ';
	while (true)
	{
		cv::Mat frame;
		cap >> frame;
		reverseX(frame);
		while (que.size() > delayFrame) que.pop();
		que.push(frame);

		//初期化
		if (loopCnt == 0) {
			unsigned char uc_zero = 0;
			bool b_zero = false;
			dmask = cv::Mat_<bool>(frame.rows + 2, frame.cols + 2, b_zero);
			result = cv::Mat(cv::Size(frame.cols, frame.rows), CV_8UC3, cv::Scalar(0, 0, 0));
		}

		//delayFrame前の画像
		cv::Mat seeFrame = que.front();

		//保存
		frames[2] = frames[1];
		frames[1] = frames[0];
		frames[0] = seeFrame;
		grays[2] = grays[1];
		grays[1] = grays[0];
		grays[0] = cv::Mat_<unsigned char>(frame.rows, frame.cols);
		cvtColor(seeFrame, grays[0], 0, 0, 0);

		//エラー対策
		loopCnt++;
		if (loopCnt < 3) continue;

		//動体検出 (1行目…マスク生成, 2～9行目…ノイズ除去など)
		doutaiMask(grays, dmask, 5);	//doutaiMaskの第2引数を大きくするほど, 「動体と判定される領域」が小さくなります
		labelingKakou(dmask, lowArea);	//ノイズ除去

		//動体だけを表示した画像
		eraseDoutai(frames[1], dmask, result);

		//停止判定, 撮影, カウント表示
		char key = cv::waitKey(1);
		if (key == 'q') break;

		if (bkey != 's' && key == 's') {
			shutterRemFrame = shutterFrame;
		}
		if (shutterRemFrame == 0) {
			shutterRemFrame = INF;
			cv::imwrite("shot.png", result);
			cv::imwrite("shot_sonomama.png", seeFrame);

			//ノイズ除去前の画像
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

		//画像表示
		imshow("SeitaiEraser", result);
		imshow("Sonomama", seeFrame);
	}
	cv::destroyAllWindows();
	return 0;
}