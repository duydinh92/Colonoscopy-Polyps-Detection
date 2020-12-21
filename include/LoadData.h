#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

using namespace std;
using namespace cv;

// Load the source image into a single float Mat of the image features
Mat loadImg(Mat img, String type) { // Type : "hist", "hog"
	Mat feature;

	if (type.compare("hist") == 0) {
		// Features vector using histogram of pixel values.
		int nVals = 255;
		float range[] = { 0, nVals };
		const float* histRange = { range };
		bool uniform = true;
		bool accumulate = false;

		calcHist(&img, 1, 0, cv::Mat(), feature, 1, &nVals, &histRange, uniform, accumulate);
		feature /= img.total();
		feature = feature.reshape(1, 1);
	}
	
	if (type.compare("hog") == 0) {
		// Feature vector using HoG
		Mat sample(64, 64, CV_32FC1);
		HOGDescriptor* hog = new HOGDescriptor(Size(64, 64), Size(32, 32), Size(16, 16), Size(16, 16), 9);
		vector<float> descriptor;

		resize(img, sample, Size(64, 64)); 
		hog->compute(sample, descriptor, Size(8, 8));

		feature = Mat(descriptor.size(), 1, CV_32FC1);
		for (int i = 0; i < descriptor.size(); i++) feature.at<float>(i) = descriptor[i];
		feature = feature.reshape(1, 1);
	}
	
	return feature;
}

// Load the image folder into train data and train labels
void loadClass(const String &dir, int label, Mat &trainData, Mat &trainLabels) {
	vector<String> files;
	glob(dir, files);

	Mat img_src, img_feature;
	for (int i = 0; i < files.size(); i++) {
		img_src = imread(files[i], IMREAD_GRAYSCALE);
		img_feature = loadImg(img_src, "hog"); // Load image feature using specific type: "hist", "hog"
		if (img_feature.empty()) continue;
		trainData.push_back(img_feature);
		trainLabels.push_back(label);
	}

	cout << trainData.rows << " " << trainData.cols << endl;
}

// Function for caculating accuracy on the test set performance (SVM)
void evaluate_svm(Mat &predicted, Mat &actual) { // Evaluate performance score for binary classification model
	assert(predicted.rows == actual.rows);
	int tp = 0; // True positive
	int fp = 0; // False positive
	int tn = 0; // True negative
	int fn = 0; // False negative

	for (int i = 0; i < actual.rows; i++) {
		float p = predicted.at<float>(i, 0);
		float a = (float) actual.at<int>(i, 0);
		if (p >= 0 && a >= 0) {
			tp++;
		}
		else if (p < 0 && a < 0) {
			tn++;
		}
		else if (p > 0 && a < 0) {
			fp++;
		}
		else fn++;
	}

	float precision = ((float) tp) / (tp + fp);
	float recall = ((float) tp) / (tp + fn);
	cout << tp << " " << tn << " " << fp << " " << fn << endl;
	cout << "Precision = " << precision << endl;
	cout << "Recall = " << recall << endl;
	cout << "F1 = " << 2 * precision * recall / (precision + recall) << endl;
	cout << "Accuracy = " << ((float)(tp + tn)) / (tp + fp + tn + fn) << endl << endl;
}

// Function for caculating accuracy on the test set performance (ANN)
void evaluate_ann(Mat& predicted, Mat& actual) { // Evaluate performance score for binary classification model
	assert(predicted.rows == actual.rows);

	int tp = 0; // True positive
	int fp = 0; // False positive
	int tn = 0; // True negative
	int fn = 0; // False negative

	for (int i = 0; i < actual.rows; i++) {
		float p = predicted.at<float>(i, 0);
		float a = (float)actual.at<int>(i, 0);
		if (p == 1 && a == 1) {
			tp++;
		}
		else if (p == 0 && a == 0) {
			tn++;
		}
		else if (p == 1 && a == 0) {
			fp++;
		}
		else fn++;
	}

	float precision = ((float)tp) / (tp + fp);
	float recall = ((float)tp) / (tp + fn);
	cout << tp << " " << tn << " " << fp << " " << fn << endl;
	cout << "Precision = " << precision << endl;
	cout << "Recall = " << recall << endl;
	cout << "F1 = " << 2 * precision * recall / (precision + recall) << endl;
	cout << "Accuracy = " << ((float)(tp + tn)) / (tp + fp + tn + fn) << endl << endl;
}
