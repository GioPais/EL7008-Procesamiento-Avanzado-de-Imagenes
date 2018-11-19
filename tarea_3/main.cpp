#define _DEBUG

// Instruciones:
// Dependiendo de la versión de opencv, pueden cambiar los archivos .hpp a usar

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/ml.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::ml;


Mat LBP(Mat img)
{
	img.convertTo(img, CV_8UC1);
	Mat output = Mat::zeros(img.rows-2, img.cols-2, CV_8UC1);
	uchar toULBP[256] ={
		1,2,3,4,5,0,6,7,8,0,0,0,9,0,10,11,12,0,0,0,0,0,0,0,13,0,0,0,14,0,
		15,16,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,0,0,0,0,0,0,0,19,
		0,0,0,20,0,21,22,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,24,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,25,0,0,0,0,0,0,0,26,0,0,0,27,0,28,29,30,31,0,32,0,0,0,33,0,
		0,0,0,0,0,0,34,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,36,37,38,0,39,0,0,0,40,0,0,0,0,0,0,0,41,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,42,43,44,0,45,0,0,0,46,0,0,0,0,0,0,0,47,48,49,0,50,
		0,0,0,51,52,53,0,54,55,56,57,58};
	for (int r = 0; r < img.rows - 2; r++) {
		for (int c = 0; c < img.cols - 2; c++) {
			int lbp_val = 0;
			Mat window = img(Rect(c, r, 3, 3));
			uchar threshold = window.at<uchar>(1, 1);
			if (threshold <= window.at<uchar>(0,0)) { lbp_val += 128;}
			if (threshold <= window.at<uchar>(0,1)) { lbp_val += 64; }
			if (threshold <= window.at<uchar>(0,2)) { lbp_val += 32; }
			if (threshold <= window.at<uchar>(1,2)) { lbp_val += 16; }
			if (threshold <= window.at<uchar>(2,2)) { lbp_val += 8; }
			if (threshold <= window.at<uchar>(2,1)) { lbp_val += 4; }
			if (threshold <= window.at<uchar>(2,0)) { lbp_val += 2; }
			if (threshold <= window.at<uchar>(1,0)) { lbp_val += 1; }

			output.at<uchar>(r,c) = toULBP[lbp_val];
		}
	}	
	return output;
}

Mat LBP2(Mat img)
{
	Mat output = Mat::zeros(img.rows - 2, img.cols - 2, CV_8UC1);
	int toULBP[256] = {
		1,2,3,4,5,0,6,7,8,0,0,0,9,0,10,11,12,0,0,0,0,0,0,0,13,0,0,0,14,0,
		15,16,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,0,0,0,0,0,0,0,19,
		0,0,0,20,0,21,22,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,24,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,25,0,0,0,0,0,0,0,26,0,0,0,27,0,28,29,30,31,0,32,0,0,0,33,0,
		0,0,0,0,0,0,34,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,36,37,38,0,39,0,0,0,40,0,0,0,0,0,0,0,41,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,42,43,44,0,45,0,0,0,46,0,0,0,0,0,0,0,47,48,49,0,50,
		0,0,0,51,52,53,0,54,55,56,57,58 };
	for (int r = 0; r < img.rows - 2; r++) {
		for (int c = 0; c < img.cols - 2; c++) {
			// Current image window
			Mat window = img(Rect(c, r, 3, 3));
			// Clockwise binary codification
			vector<int> binary_cod;
			uchar threshold = window.at<uchar>(1, 1);
			binary_cod.push_back(window.at<uchar>(0, 0) >= threshold);
			binary_cod.push_back(window.at<uchar>(0, 1) >= threshold);
			binary_cod.push_back(window.at<uchar>(0, 2) >= threshold);
			binary_cod.push_back(window.at<uchar>(1, 2) >= threshold);
			binary_cod.push_back(window.at<uchar>(2, 2) >= threshold);
			binary_cod.push_back(window.at<uchar>(2, 1) >= threshold);
			binary_cod.push_back(window.at<uchar>(2, 0) >= threshold);
			binary_cod.push_back(window.at<uchar>(1, 0) >= threshold);
			// Uniform LBP value
			int lbp_value = 0;
			for (int i = 0; i < binary_cod.size(); i++)
				lbp_value += binary_cod.at(i) * pow(2, binary_cod.size() - 1 - i);
			output.at<uchar>(r, c) = toULBP[lbp_value];
		}
	}
	return output;
}

Mat get_hist2(Mat src,String title,bool show)
{
	int histSize = 59;
	float range[] = { 0, 59 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;
	Mat hist;

	calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	//cout << "hist... " << hist << endl;
	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	
	if (show == true) {
		for (int i = 1; i < histSize; i++)
		{
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
				Scalar(256, 0, 0), 2, 8, 0);
		}

		namedWindow(title, CV_WINDOW_AUTOSIZE);
		imshow(title, histImage);
		waitKey(0);
	}

	return hist;
}

Mat get_hist(Mat src, String title, bool show)
{
	int histSize = 59;
	float range[] = { 0, 59 };
	const float* histRange = { range };


	Mat hist = Mat::zeros(1, 59, CV_32FC1);
	//cout << hist << endl;
	for (int r = 0; r < src.rows; r++) {
		for (int c = 0; c < src.cols; c++) {
			int ind = src.at<unsigned char>(r, c);
			hist.at<float>(0,ind) = hist.at<float>(0,ind) + 1;
		}
	}

	//cout << hist << endl;
	//calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	//cout << "hist... " << hist << endl;
	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = 59;

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	if (show == true) {
		for (int i = 1; i < histSize; i++)
		{
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
				Scalar(256, 0, 0), 2, 8, 0);
		}

		namedWindow(title, CV_WINDOW_AUTOSIZE);
		imshow(title, histImage);
		waitKey(0);
	}

	return hist;
}

Mat plot_hist(Mat hist,int histSize, String title)
{
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	if (true) {
		for (int i = 1; i < histSize; i++)
		{
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
				Scalar(256, 0, 0), 2, 8, 0);
		}

		//namedWindow(title, CV_WINDOW_AUTOSIZE);
		//imshow(title, histImage);
		//waitKey(0);
	}

	return histImage;
}

Mat get_features(Mat src)
{
	Mat src11 = Mat::zeros(src.rows/2, src.cols/2, CV_8UC1);
	Mat src12 = Mat::zeros(src.rows / 2, src.cols / 2, CV_8UC1);
	Mat src21 = Mat::zeros(src.rows / 2, src.cols / 2, CV_8UC1);
	Mat src22 = Mat::zeros(src.rows / 2, src.cols / 2, CV_8UC1);
	//cout << "Tamano imagen: " << src.rows << "x" << src.cols << endl;
	for (int r = 0; r < src.rows/2; r++) {
		for (int c = 0; c < src.cols/2; c++) {
			src11.at<char>(r,c) = src.at<char>(r, c);
		}
	}

	for (int r = 0; r < src.rows / 2; r++) {
		for (int c = 0; c < (src.cols/2); c++) {
			src12.at<char>(r, c) = src.at<char>(r, c+ src.cols/2);
		}
	}

	for (int r = 0; r < (src.rows/2); r++) {
		for (int c = 0; c < src.cols/2; c++) {
			src21.at<char>(r, c) = src.at<char>(r+ src.rows/2, c);
		}
	}

	for (int r = 0; r < src.rows/2; r++) {
		for (int c = 0; c < src.cols/2; c++) {
			src22.at<char>(r, c) = src.at<char>(r+src.rows/2, c+ src.cols/2);
		}
	}
	
	Mat hist11 = get_hist(src11,"hist11",false);
	Mat hist12 = get_hist(src12, "hist12",false);
	Mat hist21 = get_hist(src21, "hist21",false);
	Mat hist22 = get_hist(src22, "hist22",false);

	Mat x = Mat::zeros(1, 59*4, CV_32FC1);
	//cout << "rows: " << hist11.rows << " cols: " << hist11.cols << endl;
	//cout << "x rows: " << x.rows << " x cols: " << x.cols << endl;
	
	for (int i = 0; i < 59; i++) {x.at<float>(0,i) = hist11.at<float>(0,i);}
	for (int i = 0; i < 59; i++) { x.at<float>(0, 59+i) = hist12.at<float>(0,i); }
	for (int i = 0; i < 59; i++) { x.at<float>(0, 59*2+i) = hist21.at<float>(0,i); }
	for (int i = 0; i < 59; i++) { x.at<float>(0, 59*3+i) = hist22.at<float>(0,i); }

	return x;
}

vector<Mat> get_images(String PATH, String clase)
{
	vector<Mat> faces;
	for (int i = 0; i < 200; i++) {
		Mat rgb = imread(PATH + "separated/" + clase + "/" + to_string(i) + ".jpg");
		Mat img;
		cvtColor(rgb, img, CV_BGR2GRAY);
		faces.push_back(img);
		if (faces[i].empty()){
			cout << "Imagen no cargada" << endl;
			cvWaitKey(0);
		}
	}
	return faces;
}

vector<Mat> get_img_features(vector<Mat> images)
{
	//cout << "Extrayendo caracteristicas de imagenes..." << endl;
	vector<Mat> features;
	for (int i = 0; i < images.size(); i++) {
		//cout << "Extrayendo caracteristica de imagen " << i << endl;
		Mat image_lbp = LBP(images.at(i));
		features.push_back(get_features(image_lbp));
	}
	return features;
}

Mat build_dataset(vector<Mat> x)
{
	cout << "Generando matriz con dataset..." << endl;
	random_shuffle(x.begin(), x.end());
	Mat dataset=x.at(0);

	for (int i = 1; i < 200; i++) {
		vconcat(dataset, x.at(i), dataset);
		//features.push_back(get_features(images.at(i)));
	}
	cout << "Dimensiones matriz de datos: " << dataset.cols << "x" << dataset.rows << endl;
	return dataset;
}

vector<Mat>* split_dataset(vector<Mat> x,float size)
{
	vector<Mat> x_train;
	vector<Mat> x_test;
	int train_size = 200 * size;
	
	random_shuffle(x.begin(), x.end());

	for (int i = 1; i < x.size(); i++) {
		if (i > train_size) {
			x_test.push_back(x.at(i));
		}
		else {
			x_train.push_back(x.at(i));
		}
	}
	
	cout << "train size:"<< x_train.size() << endl;
	cout << "test size:"<< x_test.size() << endl;
	
	vector<Mat> out[2] = { x_train,x_test };
	return out;
}

float get_acc(Mat y_true, Mat y_pred) {
	float match = 0;
	for (int i = 0; i < y_true.rows; i++) {
		int true_value = y_true.at<int>(i, 0);
		int pred_value = (int)(y_pred.at<float>(i, 0));
		//cout << "y_true: " << true_value << " y_pred: " << pred_value <<endl;
		if (true_value == pred_value) { match += 1.0; }
	}
	//cout << "termine ciclo" << endl;
	return match / y_true.rows;
}

int main(void)
{
	String PATH = "C:/Projects/EL7008-Procesamiento-Avanzado-de-Imagenes/tarea_3/";
	
	vector<Mat> chinos = get_images(PATH, "Asian");
	vector<Mat> blacks = get_images(PATH, "Black");
	
	Mat chino = chinos.at(100);
	Mat black = blacks.at(100);

	if(chino.empty() || black.empty()) // No encontro la imagen
	{
		cout << "Imagen no encontrada" << endl;
		cvWaitKey(0);
		return 1;
	}
	
	imshow("chino", chino);
	imshow("black", black);
	cvWaitKey(0);

	Mat chino_LBP = LBP(chino);
	Mat black_LBP = LBP(black);

	imshow("chino_LBP", chino_LBP);
	imshow("black_LBP", black_LBP);
	cvWaitKey(0);

	Mat chino_hist = get_features(chino_LBP);
	Mat black_hist = get_features(black_LBP);

	cout << "genere las características para los 2..." << endl;

	Mat ch_hp =plot_hist(chino_hist, 236, "Chino hist");
	Mat bl_hp = plot_hist(black_hist, 236, "Black hist");
	imshow("chino_hist", ch_hp);
	imshow("black_hist", bl_hp);
	cvWaitKey(0);


	vector<Mat> x_asian = get_img_features(chinos);
	cout << "x_asian listo!" << endl;
	Mat x_0=build_dataset(x_asian);

	vector<Mat> x_black = get_img_features(blacks);
	cout << "x_black listo!" << endl;
	Mat x_1=build_dataset(x_black);
	Mat x_02, x_12;
	x_0.convertTo(x_02, CV_8UC1);
	x_1.convertTo(x_12, CV_8UC1);
	imshow("asian dataset", x_02);
	imshow("black dataset", x_12);
	cvWaitKey(0);
	//Separación de datasets
	Mat x0_train = x_0(Rect(0, 0, x_0.cols, x_0.rows * 0.7)).clone();
	Mat x0_test = x_0(Rect(0, x_0.rows * 0.7, x_0.cols, x_0.rows - x_0.rows * 0.7)).clone();
	
	Mat x1_train = x_1(Rect(0, 0, x_1.cols, x_1.rows * 0.7)).clone();
	Mat x1_test = x_1(Rect(0, x_1.rows * 0.7, x_1.cols, x_1.rows - x_1.rows * 0.7)).clone();
	
	Mat y0_train = Mat::zeros(x_0.rows*0.7, 1, CV_32SC1);
	Mat y0_test = Mat::zeros(x_0.rows-x_0.rows*0.7, 1, CV_32SC1);
	Mat y1_train = Mat::zeros(x_1.rows*0.7, 1, CV_32SC1);
	Mat y1_test = Mat::zeros(x_1.rows - x_0.rows*0.7, 1, CV_32SC1);
	for (int i = 0; i < y1_train.rows; i++) { y1_train.at<int>(i, 0) = 1; }
	for (int i = 0; i < y1_test.rows; i++) { y1_test.at<int>(i, 0) = 1; }
	//y1_train.convertTo(y1_train, CV_32SC1);
	//y1_test.convertTo(y1_test, CV_32SC1);
	//for (int i=0;)
	//cout << "train label chino" << endl;
	//cout << y0_train << endl;
	//cout << "train label negro" << endl;
	//cout << y1_train << endl;
	//cvWaitKey(0);
	//imshow("asian train", y0_train);
	//imshow("asian test", y1_test);
	//cvWaitKey(0);
	Mat x_train, x_test, y_train, y_test;
	vconcat(x0_train, x1_train, x_train);
	vconcat(x0_test, x1_test, x_test);
	vconcat(y0_train, y1_train, y_train);
	vconcat(y0_test, y1_test, y_test);


	x_train.convertTo(x_train, CV_32FC1);
	x_test.convertTo(x_test, CV_32FC1);




	// Models
	Mat y_pred, diff, nonZeroCoordinates;
	float accuracy, misses;

	// SVM
	//for (int i = 0, i < 5; i++) {

	//}
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 18, 0.001)); // -----------100 0.001
	svm->train(x_train, ROW_SAMPLE, y_train);
	svm->predict(x_test, y_pred);
	//cout << "y_test: " << y_test.flags << endl;
	//cout << y_test << endl;
	//cout << "y_pred: " << y_pred.flags << endl;
	//cout << y_pred << endl;

	//absdiff(y_pred, y_test, diff);
	//diff.convertTo(diff, CV_8UC1);
	//findNonZero(diff, nonZeroCoordinates);
	//misses = nonZeroCoordinates.total();
	//accuracy = 1 - misses / diff.size().height;
	float svm_acc = get_acc(y_test, y_pred);
	cout << "SVM Accuracy: "<< svm_acc * 100 << "%" << endl;

	// Random Forest
	Ptr<ml::RTrees> randomForest = ml::RTrees::create();
	randomForest->train(x_train, ml::ROW_SAMPLE, y_train);
	randomForest->predict(x_test, y_pred);
	//randomForest->setMaxCategories(5);
	float rf_acc = get_acc(y_test, y_pred);
	cout << "Random Forrest Accuracy: " << rf_acc * 100 << "%" << endl;

	cvWaitKey(0);
	cvWaitKey(0);

	
	return 0; // Sale del programa normalmente
}

