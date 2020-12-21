#include <opencv2/ml.hpp>
#include <LoadData.h>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main() {
	// Load training data
	cout << "Load training data ..." << endl;
	Mat trainData, trainLabels;
	loadClass("dataset_v3/train/polyps", 1, trainData, trainLabels);
	loadClass("dataset_v3/train/non_polyps", 0, trainData, trainLabels);
	trainData.convertTo(trainData, CV_32F);
	cout << endl;

	/*
	int nclasses = 2;
	int nfeatures = trainData.cols;
	Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::create();
	Mat_<int> layers(3, 1);
	layers(0) = nfeatures;     // input
	layers(1) = nclasses * 2;  // hidden 
	layers(2) = nclasses; // output, 1 pin per class.
	ann->setLayerSizes(layers);
	ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 1, 1);
	ann->setBackpropMomentumScale(0.2);
	ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 200, 0.0001));
	ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.001);

	// ann requires "one-hot" encoding of class labels:
	Mat train_classes = Mat::zeros(trainData.rows, nclasses, CV_32FC1);
	for (int i = 0; i < train_classes.rows; i++)
	{
		train_classes.at<float>(i, trainLabels.at<int>(i)) = 1.f;
	}
	ann->train(trainData, ml::ROW_SAMPLE, train_classes);
	ann->save("trained-ann-v2.xml");
	*/
	
	cout << "Load trained model ..." << endl << endl;
	Ptr<ANN_MLP> ann = Algorithm::load<ANN_MLP>("trained-ann-v2.xml");
	
	Mat output;
	Mat test = trainData.row(0);
	cout << ann->predict(test, output, true) << endl;
	cout << output.at<float>(1) << endl;
	
	
	// Load test data
	cout << "Load test data ..." << endl;
	Mat testData, testLabels;
	loadClass("dataset_v3/test/polyps", 1, testData, testLabels);
	loadClass("dataset_v3/test/non_polyps", 0, testData, testLabels);
	testData.convertTo(testData, CV_32F);
	cout << endl;

	//Evaluate model performance
	Mat predictedTrainLabels(trainLabels.rows, 1, CV_32F);
	Mat predictedTestLabels(testLabels.rows, 1, CV_32F);
	Mat sample;

	for (int i = 0; i < trainData.rows; i++) {
		sample = trainData.row(i);
		predictedTrainLabels.at<float>(i, 0) = ann->predict(sample);
	}
	for (int i = 0; i < testData.rows; i++) {
		sample = testData.row(i);
		predictedTestLabels.at<float>(i, 0) = ann->predict(sample);
	}
	
	cout << "Evaluation in train data: " << endl;
	evaluate_ann(predictedTrainLabels, trainLabels);
	cout << "Evaluation in test data: " << endl;
	evaluate_ann(predictedTestLabels, testLabels);
	
	cin.ignore();
	waitKey(0);
}