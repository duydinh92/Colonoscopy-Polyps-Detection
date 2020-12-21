#include <opencv2/ml.hpp>
#include <LoadData.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main() {
	// Load training data
	cout << "Load training data ..." << endl;
	Mat trainData , trainLabels;
	loadClass("dataset_v3/train/polyps", 1, trainData, trainLabels);
	loadClass("dataset_v3/train/non_polyps", -1, trainData, trainLabels);
	cout << endl;
	
	/*
	//Model
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setGamma(0.5);
	svm->setC(4);
	//svm->setDegree(2); //Type = Poly, Gamma = 0.5, C = 2.5
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));
	svm->train(trainData, ROW_SAMPLE, trainLabels);
	svm->save("trained-svm-v3.xml");
	cout << "Finished training process." << endl << endl;
	*/

	// Load trained model
	cout << "Load trained model ..." << endl << endl;
	Ptr<SVM> svm = Algorithm::load<SVM>("trained-svm-v3.xml");
	
	// Load test data
	cout << "Load test data ..." << endl;
	Mat testData, testLabels;
	loadClass("dataset_v3/test/polyps", 1, testData, testLabels);
	loadClass("dataset_v3/test/non_polyps", -1, testData, testLabels);
	cout << endl;
	
	//Evaluate model performance
	Mat predictedTrainLabels(trainLabels.rows, 1, CV_32FC1);
	Mat predictedTestLabels(testLabels.rows, 1, CV_32FC1);
	Mat sample;
	
	for (int i = 0; i < trainData.rows; i++) {
		sample = trainData.row(i);
		predictedTrainLabels.at<float>(i, 0) = svm->predict(sample);
	}
	for (int i = 0; i < testData.rows; i++) {
		sample = testData.row(i);
		predictedTestLabels.at<float>(i, 0) = svm->predict(sample);
	}

	cout << "Evaluation in training data: " << endl;
	evaluate(predictedTrainLabels, trainLabels);
	cout << "Evaluation in test data: " << endl;
	evaluate(predictedTestLabels, testLabels);
	
	cin.ignore();
	waitKey(0);
}