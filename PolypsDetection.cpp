#include <opencv2/ml.hpp>
#include <LoadData.h>
#include <fstream>
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::ml;


// Implement the intersection over union (IoU) between box1 and box2
float IoU(Rect box1, Rect box2) {
	// Coordinates of the intersection of box1 and box2 and caculates its area
	int xi1 = max(box1.x, box2.x);
	int yi1 = max(box1.y, box2.y);
	int xi2 = min(box1.x + box1.width, box2.x + box2.width);
	int yi2 = min(box1.y + box1.height, box2.y + box2.height);
	int inter_width = xi2 - xi1;
	int inter_height = yi2 - yi1;
	int inter_area = max(inter_width, 0) * max(inter_height, 0);

	// Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
	int box1_area = box1.width * box1.height;
	int	box2_area = box2.width * box2.height;
	int union_area = box1_area + box2_area - inter_area;

	//Return IoU
	return (float)inter_area / union_area;
}

// Find the most reasonable box for detecting polyps from set of predicted boxes
Rect chooseBox(vector <pair<Rect, float>> predicted_boxes) {
	// Caculates IoU of each pair of predicted boxes
	int n = predicted_boxes.size();
	float** iou = new float* [n]; // Array contains IoU of each pair of predicted boxes
	for (int i = 0; i < n; i++) iou[i] = new float[n];
	for (int i = 0; i < n - 1; i++) {
		for (int j = i + 1; j < n; j++) {
			iou[i][j] = IoU(predicted_boxes[i].first, predicted_boxes[j].first);
			iou[j][i] = iou[i][j];
			//cout << iou[i][j] << endl;
		}
	}

	// Caculate confidence of each predicted box and choose the one with highgest confidence
	float* conf = new float[n]; // Confidence of each predicted box
	float* other_predict = new float[n]; // Score of each predicted box which measured by other predicted boxes
	float max_conf = 0; // Max confidence
	float threshold = 0.18; // Threshold for selecting IoU of each pair predicted boxes to caculate other_predict
	int count; // Number of other predicted boxes that has IoU greater or equal than threshold
	Rect result; // Contains choosen box with highest confidence

	for (int i = 0; i < n; i++) {
		other_predict[i] = 0;
		count = 0;

		// Caculates other_predict score of i th predicted box 
		for (int j = 0; j < n; j++) {
			if (j == i) continue;
			if (iou[i][j] >= threshold) {
				other_predict[i] += iou[i][j] * (predicted_boxes[j].second);
				count++;
			}
		}
		if (count != 0) other_predict[i] = other_predict[i] / count;

		// Caculates confidence of i th predicted box 
		float bias = (float)7 / 9;
		conf[i] = (other_predict[i] * predicted_boxes[i].second) / ((bias * other_predict[i] + (1 - bias) * predicted_boxes[i].second));
		//cout << predicted_boxes[i].first.x << " " << predicted_boxes[i].first.y << " " << predicted_boxes[i].first.width << " " << predicted_boxes[i].first.height << " " << 
		//		predicted_boxes[i].second << " " << other_predict[i] << " " << conf[i] << endl;

		if (conf[i] > max_conf) {// If >= max confidence, select i th predicted box 
			max_conf = conf[i];
			result = predicted_boxes[i].first;
		}

	}

	delete[] conf;
	delete[] other_predict;
	for (int i = 0; i < n; i++) delete[] iou[i];
	delete[] iou;
	return result;
}

// Function creates bounding box with class probability
void drawPred(int left, int top, int right, int bottom, Mat& frame)
{
	// Draw bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

	/*
	// Write class probability in image
	std::string label = format("Polyps: %.3f", prob);
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
	*/
}

// Find box for polyps detection
Rect polypsDetection(String& path, Ptr<SVM>& svm, vector<pair<int, int>>& anchor) {
	// Load test image to dectect polyps
	Mat image_source = imread(path, IMREAD_GRAYSCALE);
	resize(image_source, image_source, Size(512, 512));

	// Using sliding window with predefined sizes in vector anchor boxes to detect polyps
	float max_dis = 0; // Max distance from margin of each box
	float threshold = 0.95; // remove boxes that has max distance from margin less than threshold
	int height, width; // Height and Width of each box
	const int stepSize = 10; // Step side for sliding window
	Mat window, input, output;
	Rect predicted_box, temp;
	vector <pair<Rect, float>> predicted_boxes; // Key: Predicted boxes for detecting polyps with different size, Value: max distance from margin

	for (auto& box : anchor) {
		max_dis = 0;
		height = box.first;
		width = box.second;

		for (int top = 0; top <= image_source.rows - height; top += stepSize) {
			for (int left = 0; left <= image_source.cols - width; left += stepSize) {
				temp = Rect(left, top, width, height);
				window = image_source(temp);
				input = loadImg(window, "hog");
				svm->predict(input, output, true);
				if ((output.at<float>(0) < 0) && (max_dis < abs(output.at<float>(0)))) {
					max_dis = abs(output.at<float>(0));
					predicted_box = temp;
				}
			}
		}

		if (max_dis >= threshold) predicted_boxes.push_back(make_pair(predicted_box, max_dis));
	}

	// Find the most reasonable box for detecting polyps
	predicted_box = chooseBox(predicted_boxes);

	return predicted_box;
}

int main() {
	// Vector contains set of anchor boxes with predefined sizes (height, width)
	vector<pair<int, int>> anchor = { {100, 100}, {200, 200}, {300, 300}, {400, 400},
									  {100, 200}, {200, 300}, {200, 400}, {300, 400},
									  {200, 100}, {300, 200}, {400, 200}, {400, 300} };

	// Load trained model
	Ptr<SVM> svm = Algorithm::load<SVM>("trained-svm-v3.xml");

	// Load test image to dectect polyps
	String path = "original_data\\2.jpg";


	// Show polyps detection result
	Rect predicted_box = polypsDetection(path, svm, anchor);
	Mat img = imread(path, IMREAD_COLOR);
	resize(img, img, Size(512, 512));
	drawPred(predicted_box.x, predicted_box.y, predicted_box.x + predicted_box.width, predicted_box.y + predicted_box.height, img);
	imshow("Polyps detection in the source image", img);

	waitKey();
	return 0;
}
