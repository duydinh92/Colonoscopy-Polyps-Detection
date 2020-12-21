#include <opencv2/ml.hpp>
#include <LoadData.h>
#include <fstream>
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

// Load bounding boxes coordinates from csv file
vector <Rect> loadCsv(const String& csv_file) {
	vector <Rect> actual_boxes;
	ifstream myFile(csv_file);
	if (!myFile.is_open()) throw std::runtime_error("Could not open file");
	string line, segment;
	int x[5];
	while (getline(myFile, line)) {
		stringstream stream(line);
		int i = 0;
		while (getline(stream, segment, ','))
		{
			if (i == 0) {
				i++;
				continue;
			}
			x[i] = stoi(segment);
			i++;

		}
		actual_boxes.push_back(Rect(x[1], x[2], x[3] - x[1], x[4] - x[2]));
	}
	return actual_boxes;
}

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

// Sort customed pair by second value
bool sortbysecond(const pair<Rect, float>& a, const pair<Rect, float>& b) {
	return (a.second > b.second);
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
	float threshold = 0.1;//0.7 // Threshold for selecting IoU of each pair predicted boxes to caculate other_predict
	int count; // Number of other predicted boxes that has IoU greater or equal than threshold
	vector<pair<Rect, float>> temp;

	for (int i = 0; i < n; i++) {
		other_predict[i] = 0;
		float sum_iou = 0;
		int count = 0;
		// Caculates other_predict score of i th predicted box 
		for (int j = 0; j < n; j++) {
			if (j == i) continue;
			if (iou[i][j] >= threshold) {
				other_predict[i] += iou[i][j] * (predicted_boxes[j].second);
				sum_iou += iou[i][j];
				count++;
			}
		}
		if (sum_iou != 0) other_predict[i] = other_predict[i] / sum_iou;
		// Caculates confidence of i th predicted box 
		float bias = (float)0.8;
		conf[i] = (other_predict[i] * predicted_boxes[i].second) / ((bias * other_predict[i] + (1 - bias) * predicted_boxes[i].second));
		if (conf[i] != 0 && count > 1) temp.push_back(make_pair(predicted_boxes[i].first, conf[i]));
		//cout << predicted_boxes[i].first.x << " " << predicted_boxes[i].first.y << " " << predicted_boxes[i].first.width << " " << predicted_boxes[i].first.height << " " <<
		//	predicted_boxes[i].second << " " << other_predict[i] << " " << conf[i] << endl;
	}

	Rect temp_box;
	sort(temp.begin(), temp.end(), sortbysecond);
	
	/*
	vector<Rect> result;
	for (int i = 0; i < temp.size(); i++) {
		temp_box = temp[i].first;
		result.push_back(temp_box);
		for (int j = i + 1; j < temp.size(); j++) {
			if (IoU(temp_box, temp[j].first) >= 0.5) {
				temp.erase(temp.begin() + j);
				j--;
			}
		}
		if (result.size() >= 3) break;
	}
	*/

	int max = 0;
	for (int i = 0; i < min(int(temp.size()), 5); i++) {
		if (max < temp[i].first.width * temp[i].first.height) {
			max = temp[i].first.width * temp[i].first.height;
			temp_box = temp[i].first;
		}
	}

	delete[] conf;
	delete[] other_predict;
	for (int i = 0; i < n; i++) delete[] iou[i];
	delete[] iou;
	return temp_box;
}

// Function draws bounding box with class probability
void drawPred(int left, int top, int right, int bottom, Mat& frame)
{
	// Draw bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
}

// Function draws orginal box with class probability
void drawOriginal(int left, int top, int right, int bottom, Mat& frame) {
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255));
}


// Find box for polyps detection
Rect polypsDetection(String& path, Ptr<ANN_MLP>& ann, vector<pair<int, int>>& anchor) {
	// Load test image to dectect polyps
	Mat image_source = imread(path, IMREAD_GRAYSCALE);
	resize(image_source, image_source, Size(512, 512));
	//medianBlur(image_source, image_source, 3);

	// Using sliding window with predefined sizes in vector anchor boxes to detect polyps
	float max_value = 0; // Min value for predicted box
	float threshold = 1; // remove boxes that has max distance from margin less than threshold
	int height, width; // Height and Width of each box
	const int stepSize = 10; // Step side for sliding window
	Mat window, input, output;
	Rect predicted_box, temp;
	vector <pair<Rect, float>> predicted_boxes; // Key: Predicted boxes for detecting polyps with different size, Value: max distance from margin

	double ratio = 1;
	double alpha = 1.2;
	do {
		for (auto& box : anchor) {
			max_value = 0;
			width = box.first * ratio;
			height = box.second * ratio;


			for (int top = 0; top <= image_source.rows - height; top += stepSize) {
				for (int left = 0; left <= image_source.cols - width; left += stepSize) {
					temp = Rect(left, top, width, height);
					window = image_source(temp);
					input = loadImg(window, "hog");
					ann->predict(input, output, true);
					if(output.at<float>(1) > max_value) {
						max_value = output.at<float>(1);
						predicted_box = temp;
					}
				}
			}

			if (max_value > threshold) predicted_boxes.push_back(make_pair(predicted_box, max_value));
		}

		ratio = ratio * alpha;
	} while (height * alpha <= image_source.rows && width * alpha <= image_source.cols);
	
	/*
	image_source = imread(path, IMREAD_COLOR);
	for (auto& temp : predicted_boxes) {
		predicted_box = temp.first;
		//cout << predicted_box.x << " " << predicted_box.y << " " << predicted_box.width << " " << predicted_box.height << " " << temp.second << " " << endl;
		drawPred(predicted_box.x, predicted_box.y, predicted_box.x + predicted_box.width, predicted_box.y + predicted_box.height, image_source);
	}
	imshow("All predicted boxes", image_source);
	*/
	return chooseBox(predicted_boxes);
}


// Evaluate performance of polyps detection model
void evaluate_detection(const String& image_dir, Ptr<ANN_MLP> &ann, vector<pair<int, int>> &anchor, const String& csv_file) {
	// Load files from  image directory
	vector<String> files;
	glob(image_dir, files);

	// Caculates predicted box of each image in directory
	map<int, Rect> predicted_boxes;

	vector<int> test;
	int count = 1;

	for (int i = 899; i < 999; i++) {
		predicted_boxes[stoi(files[i].substr(files[i].find_last_of("\\") + 1, files[i].find(".jpg") - files[i].find_last_of("\\") - 1))] = polypsDetection(files[i], ann, anchor);
		test.push_back(stoi(files[i].substr(files[i].find_last_of("\\") + 1, files[i].find(".jpg") - files[i].find_last_of("\\") - 1)));
		cout << count << endl;
		count++;
	}

	// Load actual boxes from csv file
	vector<Rect> actual_boxes = loadCsv(csv_file);
	//assert(actual_boxes.size() == predicted_boxes.size());
	cout << "There are total " << predicted_boxes.size() << " bounding boxes" << endl;


	// Evaluate
	int true_prediction = 0;
	for (int i = 0; i < test.size(); i++) {
		if (IoU(predicted_boxes[test[i]], actual_boxes[test[i]]) >= 0.5) true_prediction++;
	}
	cout << "Accuracy: " << (float)true_prediction / test.size();

	/*
	for (int i = 0; i < files.size(); i++) {
		predicted_boxes[stoi(files[i].substr(files[i].find_last_of("\\") + 1, files[i].find(".jpg") - files[i].find_last_of("\\") - 1))] = polypsDetection(files[i], svm, anchor);
		//cout << stoi(files[i].substr(files[i].find_last_of("\\") + 1, files[i].find(".jpg") - files[i].find_last_of("\\") - 1)) << endl;
	}

	// Load actual boxes from csv file
	vector<Rect> actual_boxes = loadCsv(csv_file);
	assert(actual_boxes.size() == predicted_boxes.size());
	cout << "There are total " << predicted_boxes.size() << " bounding boxes" << endl;

	// Evaluate
	int true_prediction = 0;
	for (int i = 0; i < actual_boxes.size(); i++) {
			if (IoU(predicted_boxes[i], actual_boxes[i]) >= 0.5) true_prediction++;
	}
	cout << "Accuracy: " << (float)true_prediction / actual_boxes.size();
	*/
}


int main() {
	// Vector contains set of anchor boxes with predefined sizes (height, width)
	//vector<pair<int, int>> anchor = { {50,100}, {100, 50}, {100, 100} };
	vector<pair<int, int>> anchor = { {100, 100} };
	// Load trained model
	Ptr<ANN_MLP> ann = Algorithm::load<ANN_MLP>("trained-ann-v2.xml");
	
	/*
	vector<Rect> actual_boxes = loadCsv("box.csv");

	// Load test image to dectect polyps
	String path = "original_data\\984.jpg";
	Rect actual_box = actual_boxes[984];
	
	
	// Show polyps detection result
	Rect predicted_box = polypsDetection(path, ann, anchor);
	cout << actual_box.x << " " << actual_box.y << " " << actual_box.width << " " << actual_box.height << endl;
	cout << predicted_box.x << " " << predicted_box.y << " " << predicted_box.width << " " << predicted_box.height << endl;
	cout << IoU(predicted_box, actual_box);
	Mat img = imread(path, IMREAD_COLOR);
	resize(img, img, Size(512, 512));
	drawPred(predicted_box.x, predicted_box.y, predicted_box.x + predicted_box.width, predicted_box.y + predicted_box.height, img);
	drawOriginal(actual_box.x, actual_box.y, actual_box.x + actual_box.width, actual_box.y + actual_box.height, img);
	imshow("Polyps detection in the source image", img);
	*/
	/*
	vector<Rect> predicted_boxes = polypsDetection(path, ann, anchor);
	for (auto& predicted_box : predicted_boxes) {
		cout << predicted_box.x << " " << predicted_box.y << " " << predicted_box.width << " " << predicted_box.height << endl;
		drawPred(predicted_box.x, predicted_box.y, predicted_box.x + predicted_box.width, predicted_box.y + predicted_box.height, img);
	}
	imshow("Polyps detection in the source image", img);
	*/
	
	String csv_file = "box.csv";
	evaluate_detection("original_data", ann, anchor, csv_file);
	cin.ignore();
	
	
	waitKey();
	return 0;
}
