#define _DEBUG
#include <iostream>
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

void ecualizar(Mat input, Mat output){
	float hist[256];
	for (int i=0; i<256; i++)
		hist[i] = 0;
	for (int r=0; r<input.rows; r++){
		for (int c=0; c<input.cols; c++){
			int ind = input.at<unsigned char>(r,c);
			hist[ind] = hist[ind]+1;
		}
	}

	float cumhist[256];
	cumhist[0]=hist[0];
	for (int i=1; i<256; i++)
		cumhist[i] = cumhist[i-1]+hist[i];

	for (int i = 0; i < 256; i++)
        cumhist[i] = floor(cumhist[i]/double(input.rows * input.cols) * 255);
    

	for (int r=0; r<input.rows; r++){
		for (int c=0; c<input.cols; c++)
			output.at<unsigned char>(r,c)=cumhist[input.at<unsigned char>(r,c)];
	}
	return;
}

int main(void)
{
	Mat originalRGB = imread("agua.png"); //Leer imagen

	if(originalRGB.empty()) // No encontro la imagen
	{
		cout << "Imagen no encontrada" << endl;
		return 1;
	}
	
	Mat original;
	cvtColor(originalRGB, original, CV_BGR2GRAY);
	
	Mat output = Mat::zeros(original.rows, original.cols, CV_8UC1);
	ecualizar(original, output);

	imshow("original", original);   // Mostrar imagen
	imwrite("original.png", original); // Grabar imagen

	imshow("ecualizado", output);   // Mostrar imagen
	imwrite("ecualizado.png", output); // Grabar imagen
	cvWaitKey(0); // Pausa, permite procesamiento interno de OpenCV

	return 0; // Sale del programa normalmente
}
