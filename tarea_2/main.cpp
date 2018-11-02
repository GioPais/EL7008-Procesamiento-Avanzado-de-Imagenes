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
	int scale = 1;
  	int delta = 0;
  	int ddepth = CV_32FC1;//CV_8UC1;
  	//int ddepth2 = CV_32FC1;
  	Mat input;

  	input.convertTo(input, ddepth);
	cvtColor(input_raw, input, CV_BGR2GRAY);

	Mat harris = Mat::zeros(input.rows, input.cols, ddepth);
	Mat input_blur = Mat::zeros(input.rows, input.cols, ddepth);


	//Mat kernel = Mat::zeros(3, 3, CV_32FC1);
	// Por hacer: calcular filtro de Harris. Para esto:
	// 1) Suavizar la imagen de entrada
	GaussianBlur(input,input_blur,Size(3,3),int(1));
	// 2) Calcular derivadas ix e iy
	Mat grad_x, grad_y;
  	Mat abs_grad_x, abs_grad_y;

  	/// Gradient X
  	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  	Sobel( input_blur, grad_x, ddepth, 1, 0, 3);
  	//convertScaleAbs( grad_x, abs_grad_x );

  	/// Gradient Y
  	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  	Sobel( input_blur, grad_y, ddepth, 0, 1, 3);
  	//convertScaleAbs( grad_y, abs_grad_y );


	// 3) Calcular momentos ixx, ixy, iyy
  	Mat ixx = grad_x.mul(grad_x); //(elemento a elemento)
	Mat ixy = grad_x.mul(grad_y); //(elemento a elemento)
	Mat iyy = grad_y.mul(grad_y); //(elemento a elemento)
	// 4) Suavizar momentos ixx, ixy, iyy
	GaussianBlur(ixx,ixx,Size(3,3),int(1));
	GaussianBlur(ixy,ixy,Size(3,3),int(1));
	GaussianBlur(iyy,iyy,Size(3,3),int(1));
	// 5) Calcular harris como: det(m) - 0.04*Tr(m)^2, con:
	//      m = [ixx, ixy; ixy, iyy]
	//      det(m) = ixx*iyy - ixy*ixy;
	//      Tr(m) = ixx + iyy
	Mat det = ixx.mul(iyy) - ixy.mul(ixy);
	Mat Tr = ixx + iyy;
	harris = det - 0.04*Tr.mul(Tr); 
	// Ademas se debe transformar la imagen de harris para que quede en el rango 0-255
	Mat output;
	normalize(harris, output, 0, 255, NORM_MINMAX, ddepth);
	output.convertTo(output, CV_8UC1);
	return output;
}

vector<KeyPoint> getHarrisPoints(Mat harris, int val)
{
	vector<KeyPoint> points;
	// Por hacer: buscar puntos de harris que sean:
	// 1) Maximos locales, y
	Mat grad_x,grad_y;
	Sobel( harris, grad_x, CV_32FC1, 1, 0, 3);
  	Sobel( harris, grad_y, CV_32FC1, 0, 1, 3);

  	int max_counter=0;
  	int hp_counter=0;

  	Mat harris_p;
  	harris.convertTo(harris_p, CV_8UC1);
  	//cout << harris;
  	//cout << "\n";

 //  	for (int r=10; r<harris.rows-10; r++){
	// 	for (int c=10 ; c<harris.cols-10; c++){
			
	// 		if (grad_x.at<float>(r,c)==0.0 && grad_y.at<float>(r,c)==0.0){
	// 			max_counter+=1;
	// 			//cout << harris.at<char>(r,c);
	// 			//cout << "\n";
	// 			if(harris.at<char>(r,c)>(char)val){
	// 				hp_counter+=1;
	// 			}
	// 		}
	// 		//input.at<float>(r,c)
	// 		//out+=input.at<float>(r,c)
			
	// 	}
	// }

  	int windows_r=20;
  	int windows_c=20;


	for (int r=0; r<harris.rows/windows_r; r++){
		for (int c=0 ; c<harris.cols/windows_c; c++){
			char h_max=0;
			int r_max;
			int c_max;
			for (int i=0; i<windows_r;i++){
				for (int j=0; j<windows_c;j++){

					if(harris.at<char>(r*windows_r+i,c*windows_c+j)>h_max){
						h_max=harris.at<char>(r*windows_r+i,c*windows_c+j);
						r_max=r*windows_r+i;
						c_max=c*windows_c+j;
					}
					
				}
			}
			if(harris.at<char>(r_max,c_max)>(char)val){
 				hp_counter+=1;
 				harris_p.at<char>(r_max,c_max)=0;
 			}
		}
	}


  	int pixeles=(harris.rows-10)*(harris.cols-10);
	
	cout << "Harris point encontrados: ";
	cout << hp_counter;
	cout << "/";
	cout << pixeles;
	cout << "\n";

  	imshow("Harris", harris);
  	imshow("Harris_point", harris_p);
	cvWaitKey(0);
	// 2) Mayores que el umbral val
	return points;
}

int main(void)
{
	Mat imleft = imread("left1.png");
	Mat imright = imread("right1.png");

	//
	imshow("original", imleft);
	//

	if(imleft.empty() || imright.empty()) // No encontro la imagen
	{
		cout << "Imagen no encontrada" << endl;
		return 1;
	}

	Mat harrisleft, harrisright;
	//cout << "hola\n";	
	harrisleft = harrisFilter(imleft);
	harrisright = harrisFilter(imright);

	imwrite("harrisleft.jpg", harrisleft); // Grabar imagen
	imwrite("harrisright.jpg", harrisright); // Grabar imagen
	imshow("harrisleft", harrisleft);
	//
	cvWaitKey(0);
	//
	//Ptr<ORB> orb = ORB::create(int nfeatures=500, float scaleFactor=1.2f,int nlevels=8,int edgeThreshold=31,int firstLevel=0,int WTA_K=2,int scoreType=ORB::HARRIS_SCORE,
	//	int  	patchSize = 31,
	//	int  	fastThreshold = 20 );
	vector<KeyPoint> pointsleft = getHarrisPoints(harrisleft, 92);
	vector<KeyPoint> pointsright = getHarrisPoints(harrisright, 112);
	//Mat descrleft, descrright;
	//orb->compute(imleft, pointsleft, descrleft);
	//orb->compute(imright, pointsright, descrright);

	// El codigo indicado arriba usa Harris para detectar puntos de interes, y luego descriptores ORB
	// Por hacer:
	// 1) Completar las funciones harrisFilter( ) y getHarrisPoints( ) indicadas arriba
	// 2) Crear imágenes que muestren los puntos de interes detectados en las dos imagenes y guardarlas
	//      Se deben llamar "impointsleft.jpg" y "impointsright.jpg"
	// 3) Hacer matching entre los descriptores de las dos imagenes, se recomienda usar:
	//      BFMatcher matcher(NORM_HAMMING);
	//      matcher.match(descrleft, descrright, matches);
	// 4) Dibujar los matches en una imagen, y guardarla, se recomienda revisar la funcion drawMatches
	//      La imagen resultante debe llamarse "img_matches.jpg"
	// 5) Crear un arreglo de puntos, y generar los pares de puntos correspondientes
	//      Se recomienda usar: vector<Point2f> points1, points2;
	//      Luego, agregar los puntos correspondientes:
	//        pointsleft[ matches[i].queryIdx ].pt
	//        pointsright[ matches[i].trainIdx ].pt
	// 6) Usar la funcion findHomography( ) para encontrar la transformación que relaciona las dos imágenes
	//      Se requiere 3 o mas correspondencias para poder calcular una homografia
	// 7) Crear una imagen imFused, la que va a almacenar las dos imagenes alineadas.
	//      Debe tener un tamano suficiente como para contener las dos imagenes alineadas
	// 8) Usar la funcion warpPerspective para proyectar la imagen imright en imFused
	// 9) Copiar los pixeles de la imagen imleft sobre imFused
	// 10) Guardar la imagen imFused

	return 0; // Sale del programa normalmente
}

