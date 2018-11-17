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

vector<KeyPoint> getHarrisPoints(Mat harris, int val)
{
	vector<KeyPoint> points;
	// Por hacer: buscar puntos de harris que sean:
	// 1) Maximos locales
	int windows_r=5;
  	int windows_c=5;

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
			// 2) Mayores que el umbral val
			if(harris.at<char>(r_max,c_max)>(char)val){
 				KeyPoint KP=KeyPoint(c_max,r_max,1);
 				points.push_back(KP);
 			}
		}
	}
  	int pixeles=(harris.rows)*(harris.cols);	
	cout << "Harris point encontrados: ";
	cout << points.size();
	cout << "/";
	cout << pixeles;
	cout << "\n";

	return points;
}

int main(void)
{
	String n;
	cin >> n;
	String img_type=".jpg";
	if(n=="1"){
		img_type=".png";
	}
	String left= "left" + n + img_type+"";
	String right= "right" + n + img_type+"";//img_type;
	Mat imleft = imread(left);
	Mat imright = imread(right);

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

	imwrite(n+"harrisleft.jpg", harrisleft); // Grabar imagen
	imwrite(n+"harrisright.jpg", harrisright); // Grabar imagen
	//imshow("harrisleft", harrisleft);
	//
	//cvWaitKey(0);
	//
	std::string const s = ("ORB"); 

	Ptr<ORB> orb = ORB::create(s);

	int vall[4]={110,115,100,100};
	int valr[4]={115,120,90,80};
	float wfactor[4]={1.60,1.9,1.5,1.4};
	float hfactor[4]={1.25,1.1,1.05,1.05};

	int idx;
	if (n=="1"){idx=0;}
	if (n=="2"){idx=1;}
	if (n=="3"){idx=2;}
	if (n=="4"){idx=3;}

	vector<KeyPoint> pointsleft = getHarrisPoints(harrisleft, vall[idx]);
	vector<KeyPoint> pointsright = getHarrisPoints(harrisright, valr[idx]);

	//-----------
	Mat l_points,r_points;
	drawKeypoints(imleft,pointsleft,l_points);
	imshow("left_points", l_points);
	drawKeypoints(imright,pointsright,r_points);
	imshow("right_points", r_points);
	cvWaitKey(0);
	//-----------
	imwrite(n+"l_points.jpg", l_points); // Grabar imagen
	imwrite(n+"r_points.jpg", r_points); // Grabar imagen

	Mat descrleft, descrright;
	orb->compute(imleft, pointsleft, descrleft);
	orb->compute(imright, pointsright, descrright);


	vector<DMatch> matches;
	BFMatcher matcher(NORM_HAMMING);
    matcher.match(descrleft, descrright, matches);
    //(4)--------------------------------------
    Mat img_matches;
    drawMatches(imleft,pointsleft,imright,pointsright,matches,img_matches);

    imshow("matches", img_matches);
	cvWaitKey(0);
	
	imwrite(n+"matches.jpg", img_matches); // Grabar imagen	

	cout << "Matches: " << matches.size() << endl;

	//FILTRAR MATCHES
	sort(matches.begin(), matches.end());
	float K=0.2;
	int KMatches = matches.size()*K;
	matches.erase(matches.begin() + KMatches, matches.end());

	cout << "Matches Filtrados: " << matches.size() << endl;
	
	drawMatches(imleft,pointsleft,imright,pointsright,matches,img_matches);

    imshow("matches_Filtrados", img_matches);
	cvWaitKey(0);
	imwrite(n+"matchesF.jpg", img_matches);
	//(5)--------------------------------------
	vector<Point2f> points1, points2;
	for (int i=0;i<matches.size();i++){
		points1.push_back(Point2f(pointsleft[ matches[i].queryIdx ].pt));
		points2.push_back(Point2f(pointsright[ matches[i].trainIdx ].pt));	
	}
	
	//cout << points1<<"\n";

	//(6)--------------------------------------
	Mat H = findHomography( points2, points1, CV_RANSAC );
	cout<< H<<"\n";


	//(7)--------------------------------------
	//Mat imFused= Mat::zeros((imleft.rows), (imleft.cols)*2, CV_32FC1);

	//(8)--------------------------------------
	Mat r_warp;
	warpPerspective(imright, r_warp, H, Size(imleft.cols*wfactor[idx],imleft.rows*hfactor[idx]));

	//imshow("r_warp", r_warp);
	//cvWaitKey(0);

	Mat imfused(r_warp.size(), CV_8UC3);
	Mat imfused_left = imfused(Rect(0, 0, imleft.cols, imleft.rows));
	Mat imfused_right = imfused(Rect(0, 0, r_warp.cols, r_warp.rows));
	r_warp.copyTo(imfused_right);
	imleft.copyTo(imfused_left);
	String PATH="/Uchile/Vision_computacional/tarea_2";
	imshow("imFused", imfused);
	cvWaitKey(0);

	imwrite(n+"alineadas.jpg", imfused); // Grabar imagen

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

