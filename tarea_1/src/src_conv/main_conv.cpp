#define _DEBUG
#define _USE_MATH_DEFINES

// Instruciones:
// Dependiendo de la versión de opencv, deben usar los primeros dos includes (cv.h, highgui.h) o bien los dos includes siguientes (imgproc.hpp, highgui.hpp)

//#include <cv.h>
//#include <highgui.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>

using namespace std;
using namespace cv;

void convolucion(Mat input, Mat mask, Mat output)
{
	// POR HACER: Programar la convolucion aca
	for (int r=mask.rows/2; r<input.rows-mask.rows/2; r++){
		for (int c=mask.cols/2 ; c<input.cols-mask.cols/2; c++){
			float out=0.0;
			for (int i=0; i<mask.rows;i++){
				for (int j=0; j<mask.cols;j++){
					out+=input.at<float>(r-mask.rows/2+i,c-mask.cols/2+j)*mask.at<float>(i,j);
				}
			}
			output.at<float>(r,c) = out;
		}
	}
}


// No es necesario modificar esta funcion
Mat fft(Mat I)
{
	Mat padded;

	int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
	Mat complexI;

	merge(planes, 2, complexI);
	
	dft(complexI, complexI);
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];
	magI += Scalar::all(1);
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols/2;
	int cy = magI.rows/2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX);
	
	Mat res;
	magI = magI*255;
	magI.convertTo(res, CV_8UC1);
	return res;
}

void Filtrar(Mat input,Mat mask){

	Mat output = input.clone();	

	convolucion(input, mask, output);

	Mat esp_out;
	esp_out = fft(output);
	imshow("spectrum_out", esp_out);
	imwrite("spectrum_out.jpg", esp_out); // Grabar imagen

	output = abs(output);

	Mat last;
	output.convertTo(last, CV_8UC1);

	imshow("filtered", last);   // Mostrar imagen
	imwrite("filtered.jpg", last); // Grabar imagen
	cvWaitKey(0); // Pausa, permite procesamiento interno de OpenCV
}

void Recto_2D(Mat input){

	// Ejemplo: filtro recto 5x5
	Mat mask = Mat(3, 3, CV_32FC1);

	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			mask.at<float>(i,j) = 1.0/9;

	Filtrar(input,mask);
}

void Recto_1D(Mat input){

	// Ejemplo: filtro recto 5x5
	Mat mask = Mat(1, 3, CV_32FC1);

	for (int i=0; i<1; i++)
		for (int j=0; j<3; j++)
			mask.at<float>(i,j) = 1.0/3;

	Mat output1 = input.clone();	

	convolucion(input, mask, output1);

	Mat mask2 = Mat(3, 1, CV_32FC1);

	for (int i=0; i<3; i++)
		for (int j=0; j<1; j++)
			mask2.at<float>(i,j) = 1.0/3;

	Filtrar(output1,mask2);
}

void Gauss_2D(Mat input){
	
	// Ejemplo: filtro recto 5x5
	Mat mask = Mat(5, 5, CV_32FC1);

	float sig=1.0;

	for (int i=0; i<5; i++){
		for (int j=0; j<5; j++){
			mask.at<float>(i,j) = exp(-(1/(2*sig*sig))*((i-mask.rows/2)*(i-mask.rows/2)+(j-mask.cols/2)*(j-mask.cols/2)))/(2*M_PI*sig*sig);
			//cout<<float out=exp(-(1/(2*sig*sig))*((i-mask.rows/2)*(i-mask.rows/2)+(j-mask.cols/2)*(j-mask.cols/2)))/(2*3.1415*sig*sig);
			//cout<< ' ';
		}
		//cout<< "\n";
	}
	Filtrar(input,mask);
}

void Gauss_1D(Mat input){
	
	// Ejemplo: filtro recto 5x5
	Mat mask = Mat(1, 5, CV_32FC1);
	float sig=1.0;
	for (int i=0; i<1; i++)
		for (int j=0; j<5; j++)
			mask.at<float>(i,j) = exp(-(1/(2*sig*sig))*((i-mask.rows/2)*(i-mask.rows/2)+(j-mask.cols/2)*(j-mask.cols/2)))/(sqrt(2*3.1415)*sig);

	Mat output1 = input.clone();	
	convolucion(input, mask, output1);
	Mat mask2 = Mat(5, 1, CV_32FC1);

	for (int i=0; i<5; i++)
		for (int j=0; j<1; j++)
			mask2.at<float>(i,j) = exp(-(1/(2*sig*sig))*((i-mask2.rows/2)*(i-mask2.rows/2)+(j-mask2.cols/2)*(j-mask2.cols/2)))/(sqrt(2*3.1415)*sig);;

	Filtrar(output1,mask2);
}

void Prewitt_v(Mat input){
	
	Mat mask = Mat(3, 3, CV_32FC1);

	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			mask.at<float>(i,j) = (j-1)/9.0;
			cout<< (j-1)/9.0;
			cout<< ' ';
		}
		cout<< "\n";
	}

	Filtrar(input,mask);
}

void Prewitt_h(Mat input){
	
	Mat mask = Mat(3, 3, CV_32FC1);

	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			mask.at<float>(i,j) = (i-1)/9.0;
			cout<< (i-1)/9.0;
			cout<< ' ';
		}
		cout<< "\n";
	}	

	Filtrar(input,mask);
}

void Laplaciano(Mat input){
	
	Mat mask = Mat(3, 3, CV_32FC1);

	for (int i=0; i<3; i++){
		for (int j=0; j<3; j++){
			if(i==1 & j==1){
				mask.at<float>(i,j) = 8/9.0;
				cout<< 8/9.0;
			}
			else{
				mask.at<float>(i,j) = -1/9.0;
				cout<< -1/9.0;
			}
			cout<< ' ';
		}
		cout<< "\n";
	}

	Filtrar(input,mask);
}

void log_filter(Mat original){
	Mat input;
	original.convertTo(input, CV_32FC1);
	
	// Ejemplo: filtro recto 5x5
	Mat mask = Mat(5, 5, CV_32FC1);

	float sig=1.0;

	for (int i=0; i<5; i++){
		for (int j=0; j<5; j++){
			float d = (i-mask.rows/2)*(i-mask.rows/2)+(j-mask.cols/2)*(j-mask.cols/2);
			float e = exp(-(1/(2*sig*sig))*d);
			mask.at<float>(i,j) = (-1/(M_PI*sig*sig*sig*sig))*(1-(d/(2*sig*sig)))*e;
			//cout<<exp(-(1/(2*sig*sig))*((i-mask.rows/2)*(i-mask.rows/2)+(j-mask.cols/2)*(j-mask.cols/2)))/(2*3.1415*sig*sig);
			cout<< ' ';
		}
		cout<< "\n";
	}

	Filtrar(input,mask);
}

void log_filter2(Mat original){
	Mat input;
	original.convertTo(input, CV_32FC1);
	
	// Ejemplo: filtro recto 5x5
	Mat mask = Mat(5, 5, CV_32FC1);
	float sig=1.0;
	for (int i=0; i<5; i++)
		for (int j=0; j<5; j++)
			mask.at<float>(i,j) = exp(-(1/(2*sig*sig))*((i-mask.rows/2)*(i-mask.rows/2)+(j-mask.cols/2)*(j-mask.cols/2)))/(2*3.1415*sig*sig);

	Mat output1 = input.clone();	

	convolucion(input, mask, output1);

	Mat mask2 = Mat(5, 5, CV_32FC1);

	for (int i=0; i<5; i++){
		for (int j=0; j<5; j++){
			if(i==2 & j==2){
				mask2.at<float>(i,j) = 24/25.0;
				cout<< 24/25.0;
			}
			else{
				mask2.at<float>(i,j) = -1/25.0;
				cout<< -1/25.0;
			}
			cout<< ' ';
		}
		cout<< "\n";
	}

	Filtrar(input,mask);
}

int main(void)
{
	Mat originalRGB = imread("figuras.png"); //Leer imagen

	if(originalRGB.empty()) // No encontro la imagen
	{
		cout << "Imagen no encontrada" << endl;
		return 1;
	}
	
	Mat original;
	cvtColor(originalRGB, original, CV_BGR2GRAY);

	imshow("original", original);   // Mostrar imagen
	//imwrite("original.jpg", input); // Grabar imagen
	//---------------------------

	Mat input;
	original.convertTo(input, CV_32FC1);

	Mat esp_in;
	esp_in = fft(input);
	imshow("spectrum_in", esp_in);
	imwrite("spectrum_in.jpg", esp_in); // Grabar imagen
	
	Recto_2D(input);
	Recto_1D(input);
	Gauss_2D(input);
	//Gauss_1D(input);

	Prewitt_v(input);
	cout<< "--------------------\n";
	Prewitt_h(input);
	cout<< "--------------------\n";
	Laplaciano(input);
	cout<< "--------------------\n";
	log_filter(input);

	return 0; // Sale del programa normalmente
}
