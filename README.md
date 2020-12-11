# Colonoscopy-Polyps-Detection
Developed model for detecting polyps in colonoscopy images

## Requirements
Install OpenCV C++ in Visual Studio project.

## 1. Dataset
1. *original_data*: 1000 images
2. *dataset_v3* (contain cropped polyps and non_polyps images from *original_data*) : 
    * *train*: polyps (750 images) and non_polyps (2400 images)
    * *test*: polyps (250 images) and non_polyps (800 images)
3. *box.csv*: contains coordinates of polyps image "i.jpg" in the original image, use for evaluating model performance (format: i, left, top, width, height)
 
 ## 2. Model
 1. *include/LoadData.h*: Load the source image into features (using HoG) and load the image folder into train data and train labels.
 2. *src/Model.cpp*: Build classifiers for polyps/non-polyps images using SVM (Support Vector Machine) with RBF Kernel.
 ```sh
Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::RBF);
	svm->setGamma(0.5);
	svm->setC(4);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, (int)1e7, 1e-6));
	svm->train(trainData, ROW_SAMPLE, trainLabels);
	svm->save("trained-svm-v3.xml");
```
 3. *src/PolypsDetection.cpp*: Using sliding window and pretrained classifier to detect polyps in colonoscopy images
 Load pretrained model
 ```sh
Ptr<SVM> svm = Algorithm::load<SVM>("trained-svm-v3.xml");
```
Demo
 ```sh

```
 4. *src/EvaluatePerformance.cpp*: evaluating model performance
 
