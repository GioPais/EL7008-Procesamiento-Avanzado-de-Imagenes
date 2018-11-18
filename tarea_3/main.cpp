#define _DEBUG

// Instruciones:
// Dependiendo de la versión de opencv, pueden cambiar los archivos .hpp a usar

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>

using namespace std;
using namespace cv;

Mat harrisFilter(Mat input_raw)
{
	int ddepth = CV_32FC1;
  	Mat input;

  	//input.convertTo(input, ddepth);
	cvtColor(input_raw, input, CV_BGR2GRAY);

	//Mat harris = Mat::zeros(input.rows, input.cols, ddepth);

	// Por hacer: calcular filtro de Harris. Para esto:
	// 1) Suavizar la imagen de entrada
	Mat input_blur;
	GaussianBlur(input,input_blur,Size(3,3),1);
	// 2) Calcular derivadas ix e iy
	Mat grad_x, grad_y;
  	/// Gradient X
  	Sobel( input_blur, grad_x, ddepth, 1, 0, 3);
  	/// Gradient Y
  	Sobel( input_blur, grad_y, ddepth, 0, 1, 3);
  	
	// 3) Calcular momentos ixx, ixy, iyy
  	Mat ixx = grad_x.mul(grad_x); //(elemento a elemento)
	Mat ixy = grad_x.mul(grad_y); //(elemento a elemento)
	Mat iyy = grad_y.mul(grad_y); //(elemento a elemento)
	// 4) Suavizar momentos ixx, ixy, iyy
	GaussianBlur(ixx,ixx,Size(3,3),1);
	GaussianBlur(ixy,ixy,Size(3,3),1);
	GaussianBlur(iyy,iyy,Size(3,3),1);
	// 5) Calcular harris como: det(m) - 0.04*Tr(m)^2, con:
	Mat det = ixx.mul(iyy) - ixy.mul(ixy);
	Mat Tr = ixx + iyy;
	Mat harris = det - 0.04*Tr.mul(Tr); 
	// Ademas se debe transformar la imagen de harris para que quede en el rango 0-255
	Mat harris_n;
	normalize(harris, harris_n, 0, 255, NORM_MINMAX, ddepth);
	harris_n.convertTo(harris_n, CV_8UC1);
	return harris_n;
}


Mat LBP(Mat img)
{
	Mat output = Mat::zeros(img.rows, img.cols, CV_8UC1);
	int toULBP[256] ={
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
	for (int r = 1; r < img.rows - 1; r++) {
		for (int c = 1; c < img.cols - 1; c++) {
			int code_val = 0;
			if (img.at<uchar>(r, c) <= img.at<uchar>(r - 1, c - 1)) { code_val += 128;}
			if (img.at<uchar>(r, c) <= img.at<uchar>(r - 1, c)) { code_val += 64; }
			if (img.at<uchar>(r, c) <= img.at<uchar>(r - 1, c+1)) { code_val += 32; }
			if (img.at<uchar>(r, c) <= img.at<uchar>(r, c + 1)) { code_val += 16; }
			if (img.at<uchar>(r, c) <= img.at<uchar>(r + 1, c + 1)) { code_val += 8; }
			if (img.at<uchar>(r, c) <= img.at<uchar>(r + 1, c)) { code_val += 4; }
			if (img.at<uchar>(r, c) <= img.at<uchar>(r + 1, c - 1)) { code_val += 2; }
			if (img.at<uchar>(r, c) <= img.at<uchar>(r, c - 1)) { code_val += 8; }

			output.at<char>(r-1, c-1) = (char)toULBP[code_val];
		}
	}	
	return output;
}

Mat get_hist(Mat src,String title)
{
	int histSize = 59;
	float range[] = { 0, 59 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;
	Mat hist;

	calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);
	cout << "hist... " << hist << endl;
	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(256, 0, 0), 2, 8, 0);
	}

	namedWindow(title, CV_WINDOW_AUTOSIZE);
	imshow(title, histImage);
	waitKey(0);

	return hist;
}

Mat get_features(Mat src)
{
	Mat src11 = Mat::zeros(src.rows/2, src.cols/2, CV_8UC1);
	Mat src12 = Mat::zeros(src.rows / 2, src.cols / 2, CV_8UC1);
	Mat src21 = Mat::zeros(src.rows / 2, src.cols / 2, CV_8UC1);
	Mat src22 = Mat::zeros(src.rows / 2, src.cols / 2, CV_8UC1);

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
	
	Mat hist11 = get_hist(src11,"hist11");
	Mat hist12 = get_hist(src12, "hist12");
	Mat hist21 = get_hist(src21, "hist21");
	Mat hist22 = get_hist(src22, "hist22");

	Mat x = Mat::zeros(1, 59*4, CV_32FC1);
	cout << "rows: " << hist11.rows << " cols: " << hist11.cols << endl;
	cout << "x rows: " << x.rows << " x cols: " << x.cols << endl;
	
	for (int i = 0; i < 59 ; i++) {x.at<float>(0,i) = hist11.at<float>(i,0);}
	for (int i = 0; i < 59; i++) { x.at<float>(0, 59+i) = hist12.at<float>(i, 0); }
	for (int i = 0; i < 59; i++) { x.at<float>(0, 59*2+i) = hist21.at<float>(i, 0); }
	for (int i = 0; i < 59; i++) { x.at<float>(0, 59*3+i) = hist22.at<float>(i, 0); }

	return x;
}


int main(void)
{
	String PATH = "C:/Projects/EL7008-Procesamiento-Avanzado-de-Imagenes/tarea_3/";
	String n = "1";
	String clase = "Asian";
	Mat chino_rgb = imread(PATH+"separated/"+clase+"/"+n+".jpg");
	//Mat imright = imread(right);
	clase = "Black";
	Mat black_rgb = imread(PATH + "separated/" + clase + "/" + n + ".jpg");
	
	Mat chino; Mat black;
	cvtColor(chino_rgb, chino, CV_BGR2GRAY);
	cvtColor(black_rgb, black, CV_BGR2GRAY);

	if(chino.empty() || black.empty()) // No encontro la imagen
	{
		cout << "Imagen no encontrada" << endl;
		cvWaitKey(0);
		return 1;
	}
	
	imshow("chino", chino);
	cvWaitKey(0);

	imshow("black", black);
	cvWaitKey(0);

	Mat chino_LBP = LBP(chino);
	Mat black_LBP = LBP(black);

	imshow("chino_LBP", chino_LBP);
	cvWaitKey(0);

	imshow("black_LBP", black_LBP);
	cvWaitKey(0);

	Mat chino_hist = get_features(chino_LBP);
	Mat black_hist = get_features(black_LBP);

	imshow("chino_hist", chino_hist);
	cvWaitKey(0);

	imshow("black_hist", black_hist);
	cvWaitKey(0);


	
	return 0; // Sale del programa normalmente
}

