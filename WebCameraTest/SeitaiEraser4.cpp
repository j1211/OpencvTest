//・3フレーム間差分 (色が変わった画素はそのまま, 色が変わらない画素は黒く表示）
//・ラベリング処理（ノイズ除去, 黒い穴を埋める処理)
//今回追加！
//・ディレイバー(フレーム数で指定)
//・シャッター機能 (キーボードのsボタンを押してから, shutter_frameフレーム後に写真を撮る)

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <queue>
#include <algorithm>
using namespace std;

int Abs(int x) { if (x > 0) return x; return -x; }

//輪郭
cv::Mat_<unsigned char> rinkaku(cv::Mat_<unsigned char> gray, int sikii) {
	cv::Mat_<unsigned char> ret(gray.rows, gray.cols);
	int filter[3][3] = { { 0, -1, 0 },{ -1, 4, -1 },{ 0, -1, 0 } };

	for (int y = 0; y < gray.rows; y++) {
		for (int x = 0; x < gray.cols; x++) {
			int diff = 0;
			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					int sy = y + dy;
					int sx = x + dx;
					if (sy == 0 || sy + 1 == gray.rows || sx == 0 || sx + 1 == gray.cols) continue;

					diff += gray(sy, sx) * filter[dy + 1][dx + 1];
				}
			}
			diff = abs(diff);
			if (diff >= sikii) ret[y][x] = 255;
			else ret[y][x] = 0;
		}
	}
	return ret;
}

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
cv::Mat_<unsigned char> cvtColor(cv::Mat frame, double weightR, double weightG, double weightB) {
	cv::Mat_<unsigned char> ret(frame.rows, frame.cols);

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
			ret(y, x) = newColor;
		}
	}
	return ret;
}

//動体検出 (動体部分なら255(白), それ以外なら0(黒), にした2値画像)
cv::Mat_<unsigned char> doutaiMask(cv::Mat_<unsigned char> gray[3], int sikii = 1) {
	int h = gray[0].rows;
	int w = gray[0].cols;
	cv::Mat_<unsigned char> mask(h, w);

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			if (Abs(gray[0](y, x) - (int)gray[1](y, x)) >= sikii && Abs(gray[1](y, x) - (int)gray[2](y, x)) >= sikii) {
				mask[y][x] = 255;
			}
			else {
				mask[y][x] = 0;
			}
		}
	}
	return mask;
}


//番兵法
//入れる
cv::Mat_<unsigned char> makeBanhei(cv::Mat_<unsigned char> dmask) {
	int h = dmask.rows;
	int w = dmask.cols;
	int initValue = 0;
	cv::Mat_<unsigned char> figure(h + 2, w + 2, initValue);

	for (int y = 1; y <= h; y++) {
		for (int x = 1; x <= w; x++) {
			figure[y][x] = dmask[y - 1][x - 1];
		}
	}
	return figure;
}

//取る
cv::Mat_<unsigned char> eraseBanhei(cv::Mat_<unsigned char> dmask) {
	int h = dmask.rows;
	int w = dmask.cols;
	cv::Mat_<unsigned char> figure(h - 2, w - 2);
	for (int y = 0; y < h - 2; y++) {
		for (int x = 0; x < w - 2; x++) {
			figure[y][x] = dmask[y + 1][x + 1];
		}
	}
	return figure;
}

//ラべリング(0番から)
pair<int, cv::Mat_<int>> labeling(cv::Mat_<unsigned char> dmask) {
	int h = dmask.rows;
	int w = dmask.cols;
	const int dy[4] = { -1, 0, 1, 0 };
	const int dx[4] = { 0, 1, 0, -1 };
	int initValue = -1;
	cv::Mat_<int> ret(h, w, initValue);
	queue<int> que;
	int id = 0;

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			if (ret[y][x] != -1) continue;

			unsigned char val = dmask[y][x];

			que.push(y);
			que.push(x);
			ret[y][x] = id;
			while (!que.empty()) {
				int y = que.front(); que.pop();
				int x = que.front(); que.pop();
				for (int i = 0; i < 4; i++) {
					int ny = y + dy[i];
					int nx = x + dx[i];
					if (0 <= ny && ny < h && 0 <= nx && nx < w && ret[ny][nx] == -1 && dmask[ny][nx] == val) {
						ret[ny][nx] = id;
						que.push(ny);
						que.push(nx);
					}
				}
			}
			id++;
		}
	}
	return pair<int, cv::Mat_<int>>(id, ret);
}

//各ラベルの座標集合(y, x)
vector<vector<pair<int, int>>> getLabelUnion(pair<int, cv::Mat_<int>> label) {
	vector<vector<pair<int, int>>> uni(label.first);
	for (int y = 0; y < label.second.rows; y++) {
		for (int x = 0; x < label.second.cols; x++) {
			uni[label.second[y][x]].push_back(pair<int, int>(y, x));
		}
	}
	return uni;
}

//マスク加工
//面積lowArea以下の「白(255)」を削除.
void eraseNoiz(cv::Mat_<unsigned char> dmask, vector<vector<pair<int, int>>> &labelUnion, int lowArea) {
	for (int id = 0; id < labelUnion.size(); id++) {
		if (dmask[labelUnion[id][0].first][labelUnion[id][0].second] > 0 && labelUnion[id].size() <= lowArea) {
			//削除処理
			for (int i = 0; i < labelUnion[id].size(); i++) {
				dmask[labelUnion[id][i].first][labelUnion[id][i].second] = 0;
			}
		}
	}
}
//0番以外の「黒(0)」を白くする
void blackToWhite(cv::Mat_<unsigned char> dmask, vector<vector<pair<int, int>>> &labelUnion) {
	for (int id = 1; id < labelUnion.size(); id++) {
		if (dmask[labelUnion[id][0].first][labelUnion[id][0].second] == 0) {
			//白くする処理
			for (int i = 0; i < labelUnion[id].size(); i++) {
				dmask[labelUnion[id][i].first][labelUnion[id][i].second] = 255;
			}
		}
	}
}

//表示画像の生成 (動体はカラーで, それ以外はデフォルト色で)
cv::Mat eraseDoutai(cv::Mat frame, cv::Mat_<unsigned char> doutaiMask, int defaultR = 0, int defaultG = 0, int defaultB = 0) {
	int h = frame.rows;
	int w = frame.cols;
	cv::Mat result(cv::Size(w, h), CV_8UC3);

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			if (doutaiMask(y, x) == 0) {	//動体
				result.at<cv::Vec3b>(y, x) = cv::Vec3b(defaultB, defaultG, defaultR);
			}
			else {	//それ以外
				result.at<cv::Vec3b>(y, x) = frame.at<cv::Vec3b>(y, x);
			}
		}
	}
	return result;
}

//本体
cv::Mat frames[3];	//frames[0]が一番新しい
cv::Mat_<unsigned char> grays[3];
queue<cv::Mat> que;

int main()
{
	cv::VideoCapture cap(0);
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

		//delayFrame前の画像
		cv::Mat seeFrame = que.front();

		//保存
		frames[2] = frames[1];
		frames[1] = frames[0];
		frames[0] = seeFrame;
		grays[2] = grays[1];
		grays[1] = grays[0];
		grays[0] = cvtColor(seeFrame, 0, 0, 0);

		//エラー対策
		loopCnt++;
		if (loopCnt < 3) continue;

		//動体検出 (1行目…マスク生成, 2～9行目…ノイズ除去など)
		cv::Mat_<unsigned char> dmask = doutaiMask(grays, 5);	//doutaiMaskの第2引数を大きくするほど, 「動体と判定される領域」が小さくなります
		cv::Mat_<unsigned char> dmaskB = makeBanhei(dmask);
		pair<int, cv::Mat_<int>> labelFigure = labeling(dmaskB);
		vector<vector<pair<int, int>>> labelUnion = getLabelUnion(labelFigure);
		eraseNoiz(dmaskB, labelUnion, lowArea);
		labelFigure = labeling(dmaskB);
		labelUnion = getLabelUnion(labelFigure);
		blackToWhite(dmaskB, labelUnion);
		cv::Mat_<unsigned char> extra_dmask = eraseBanhei(dmaskB);

		//動体だけを表示した画像
		cv::Mat result = eraseDoutai(frames[1], extra_dmask);

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
			cv::imwrite("shot_noizExist.png", eraseDoutai(frames[1], dmask));
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