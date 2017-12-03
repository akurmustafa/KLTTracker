
//Sample for loading and displaying an image(RGB)

/*
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;


int main(){
	
	Mat image;
	string imgName = "lena.jpg";
	image = imread(imgName, CV_LOAD_IMAGE_COLOR);	//READ THE FÝLE

	if (!image.data){
		cout << "Couldn't open or find the image" << std::endl;
		return -1;
	}
	namedWindow("Display Window", WINDOW_AUTOSIZE);	//crate a window for display

	imshow("Display Window", image);		//show our image inside it

	waitKey(0);		//Wait for a keystroke in the window
	return 0;

}
*/


//Sample for loading and displaying an image(gray scale)

/*
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;


int main(){

	Mat image;
	string imgName = "lena.jpg";
	image = imread(imgName, CV_LOAD_IMAGE_GRAYSCALE);	//READ THE FÝLE

	if (!image.data){
		cout << "Couldn't open or find the image" << std::endl;
		return -1;
	}
	namedWindow("Display Window", WINDOW_AUTOSIZE);	//crate a window for display

	imshow("Display Window", image);		//show our image inside it

	waitKey(0);		//Wait for a keystroke in the window
	return 0;
}

*/


//Sample for loading and displaying an image(LOADS THE IMAGE AS IT IS INCLUDING THE ALPHA CHANNEL IF PRESENT)

/*
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;


int main(){

	Mat image;
	string imgName = "lena.jpg";
	image = imread(imgName, CV_LOAD_IMAGE_UNCHANGED);	//READ THE FÝLE

	if (!image.data){
		cout << "Couldn't open or find the image" << std::endl;
		return -1;
	}
	namedWindow("Display Window", WINDOW_AUTOSIZE);	//crate a window for display

	imshow("Display Window", image);		//show our image inside it

	waitKey(0);		//Wait for a keystroke in the window
	return 0;

}

*/

//Sample to load image, convert image, and save them to the disk

/*
#include<opencv2/opencv.hpp>
//#include<opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include <cv.h>
//#include <highgui.h>
#include <iostream>
using namespace cv;
using namespace std;
int main(){
	char * imageName = "lena.jpg";
	Mat image;
	image = imread(imageName, 1);
	if (!image.data){
		cout << "Couldn't open or find the image" << std::endl;
		return -1;
	}
	Mat grayImage;
	cvtColor(image,grayImage,CV_BGR2GRAY);
	imwrite("./images/grayImage.jpg",grayImage);

	namedWindow(imageName, CV_WINDOW_AUTOSIZE);
	namedWindow("Gray Image",CV_WINDOW_AUTOSIZE);

	imshow(imageName, image);
	imshow("Gray Image",grayImage);

	waitKey(0);
	return 0;
}
*/


//Sample load image, copy image, create a new image that is subsection of the image IMPORTANT

/*
#include<opencv2/opencv.hpp>
//#include<opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include <cv.h>
//#include <highgui.h>
#include <iostream>
using namespace cv;
using namespace std;
int main(){
	char * imageName = "lena.jpg";
	Mat A;
	A = imread(imageName, CV_LOAD_IMAGE_COLOR);
	//cout << A << endl;
	Mat B = A.clone();
	imwrite("./images/copyLena.jpg", B);

	//Rect 0,0 is the position of the upper left pixel o the rect 100(horizontal),100(vertical) are sizes. 
	Mat D(A,Rect(0,0,1,2));
	Mat E(A, Rect(2, 2, 1, 2));
	Mat F = imread("./images/lenaDar.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	//Mat M(117, 139, CV_8UC3, Scalar(0, 0, 255));
	//Mat M(1, 2, CV_32F);
	//Mat N(117, 139, CV_32F);
	Mat N;
	//N.create(117, 139, CV_8UC(2));
	Mat M = 5*Mat::ones(139, 117, CV_32F);
	multiply(F,M,N,1,CV_32S);
	cout << N;
	//randu(M, Scalar::all(0), Scalar::all(4096));
	namedWindow("N", WINDOW_AUTOSIZE);
	imshow("N", N);
	//M = D * 2;
	//cout << F << endl;
	//cout << M << endl;
	//cout << N << endl;
	//Mat E = A(Range::all(),Range(5,7));		//ask to someone
	char * imagePosD = "./images/subsectionLena.jpg";
	imwrite(imagePosD,D);

	char * imagePosB = "./images/copyOfLena.jpg";
	imwrite(imagePosB,B);
	namedWindow(imagePosB,WINDOW_AUTOSIZE);
	imshow(imagePosB, B);

	namedWindow(imagePosD,WINDOW_AUTOSIZE);
	imshow(imagePosD,D);

	waitKey(0);
	
	return 0;
}
*/


//sample for matrix creation there are some other methods if needed lookup to the tutorial

/*
#include<opencv2/opencv.hpp>
//#include<opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include <cv.h>
//#include <highgui.h>
#include <iostream>
using namespace cv;
using namespace std;

int main(){

	Mat M(2,2,CV_8UC3,Scalar(0,0,255));
	cout << "M" << endl << " " << M << endl << endl;

	M.create(4,4,CV_8UC(2));
	cout << "M" << endl << " " << M << endl << endl;

	//namedWindow("test window", WINDOW_AUTOSIZE);
	//imwrite("test screen",M);
	while(true){

	}
	//waitKey(0);
	return 0;
}
*/


//eksikler var sharpening function

/*
#include<opencv2/opencv.hpp>
//#include<opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include <cv.h>
//#include <highgui.h>
#include <iostream>
using namespace cv;
using namespace std;

void Sharpen(const Mat& myImage,Mat& Result){
	CV_Assert(myImage.depth==CV_8U);	//accept only uchar images
	Result.create(myImage.size(), myImage.type());
	const int nchannels = myImage.channels;
	
	for (int j = 1; j < myImage.rows; j++){
		const uchar* previous = myImage.ptr<uchar>(j-1);
		const uchar* current = myImage.ptr<uchar>(j);
		const uchar* next = myImage.ptr<uchar>(j+1);

		uchar* output = Result.ptr<uchar>(j);

		for (int i = nchannels; i < nchannels*(myImage.cols-1); i++){
			*output++ = saturate_cast<uchar>(5*current[i]-current[i-nchannels]-current[i+nchannels]
				-previous[i]-next[i]);
		}
	}
	Result.row(0).setTo(Scalar(0));
	Result.row(Result.rows - 1).setTo(Scalar(0));
	Result.col(0).setTo(Scalar(0));
	Result.col(Result.cols - 1).setTo(Scalar(0));


}
int main(){

	int divideWith;	//convert out input string to the integer
	stringstream s;
	s << "2";
	s >> divideWith;
	if (!s || !divideWith){
		cout << "Input is not valid" << endl;
		return -1;
	}
	double t = (double)getTickCount();
	uchar table[256];
	for (int i = 0; i < 256; i++){
		table[i] = (uchar)(divideWith)*(i / divideWith);
	}
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "time passed is in seconds " << t << endl;
	while (true){

	}
	//waitKey(0);
	return 0;
}
*/


//filter2d function implementation(deneme)

/*
#include<opencv2/opencv.hpp>
//#include<opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include <cv.h>
//#include <highgui.h>
#include <iostream>
using namespace cv;
using namespace std;

int main(){
	char* imageName = "lena.jpg";
	Mat im1;
	im1= imread(imageName,CV_LOAD_IMAGE_COLOR);
	Mat kern = (Mat_<char>(3, 3) << 0, -1, 0,
									-1, 5, -1,
									0, -1, 0);
	Mat im2 = im1.clone();
	filter2D(im1, im2, im1.depth(), kern);
	//Mat Result;
	imwrite("./images/filteredImage.jpg",im2);
	namedWindow("Filtered Image",WINDOW_AUTOSIZE);
	imshow("Filtered Image",im2);
	waitKey(0);



	return 0;
}
*/


//file creating opening

/*
#include<iostream>
#include<string>
#include<fstream>
using namespace std;
int main(){
	//ofstream constructor opens a file
	ofstream outClientFile("clients.txt",ios::out);
	if (!outClientFile)	//overloaded ! operator
	{
		cerr << "File couldn't be opened" << endl;
		exit(1);
	}	//end if
	cout << "Enter the account, name, and balance" << endl<<"Enter end-of-file to end input.\n?";
	int account;
	string name;
	double balance;
	//read account, name, and balance from cin, then place in file
	while (cin >> account >> name>>balance){
		outClientFile << account << " "<< name<<" " << balance << endl;
		cout << "? ";
	}	//end while
	return 0;
}
*/


//Input from txt

/*
#include<iostream>
#include<string>
#include<fstream>
#include<iomanip>
#include<cstdlib>
using namespace std;

void outputLine(int,const string,double);	//prototype

int main(){
	//ifstream constructor opens the file
	ifstream inClientFile("clients.txt", ios::in);

	//exit program if ifstream couldn't open the file
	if (!inClientFile){
		cerr << "cannot open the file" << endl;
		exit(1);
	}//end if

	int account;
	string name;
	double balance;
	cout << left << setw(10) << "Account" << setw(13) << "Name" << "balance" << endl << fixed << showpoint;

	//display each record in file
	int lineCounter =0 ;
	string line;
	while (getline(inClientFile,line)){
		lineCounter++;
	}
	inClientFile.close();
	ifstream file("clients.txt", ios::in);
	cout <<"Number of lines are "<< lineCounter <<endl;
	for (int i = 0; i < lineCounter; i++){
	//while (!EOF){	//doesn't work
		file >> account >> name >> balance;
		outputLine(account, name, balance);
	}
	file.close();
	while (true){

	}

	return 0;
}
//display single record from file
void outputLine(int account,const string name,double balance){
	cout << left << setw(10) << account << setw(13) << name << setw(7) << setprecision(2) << right<<balance << endl;
}
*/


//function to read matrix data from txt file

/*
#include<iostream>
#include<string>
#include<fstream>
#include<iomanip>
#include<cstdlib>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat readDataFromTxt(string fileName,int rows,int cols){
	double m;
	Mat out = Mat::zeros(rows,cols,CV_64FC1);

	ifstream fileStream(fileName);
	int cnt = 0;//index starts from the 0
	while (fileStream>>m){
		int tempRow = cnt / cols;
		int tempCol = cnt%cols;
		out.at<double>(tempRow, tempCol) = m;
		cnt++;
	}
	return out;
}
int main(){
	string fileName = "imageData.txt";
	int rowNum = 3;
	int colNum = 3;

	Mat A= readDataFromTxt(fileName, rowNum, colNum);
	cout << "A= " << A << endl;
	namedWindow("A", WINDOW_AUTOSIZE);
	imshow("A",A);
	while (true){

	}
	/*
	string imName = "lena.jpg";
	Mat C = imread(imName, CV_LOAD_IMAGE_COLOR);
	cout << C.type() << endl << C.size() << endl;
	//double arr1[9] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	Mat A = (Mat_<double>(3, 3) << 4000, 4000, 2000, 4000, 4000, 4000, 4000, 4000, 4000);
	//A.row(0) = (Mat_<double>(3, 1)<< 1, 2, 10);
	Mat cop = (Mat_<double>(1, 3) << 4, 5, 6);
	//A.row(0).assignTo(cop,-1);
	A.row(0) = cop.clone();
	A.at<double>(0,0)=1;
	//A.row(1) = (Mat_<double>(1, 3)<< 4, 5, 6);
	//A.row(2) = (Mat_<double>(1, 3)<< 7, 8, 9);
	//cout << A.type() << endl << A.size() << endl;
	cout << "A= " << endl << " " << A << endl;
	namedWindow("try", WINDOW_AUTOSIZE);
	imshow("try", A);

	//namedWindow("try",WINDOW_AUTOSIZE);	
	//imshow("try",A);
	//cout << A.col(1)<<endl;

	Mat_<Vec3b> img(100, 100,Vec3b(0,255,0));
	cout << img.type()<<endl<<img.size() << endl;
	namedWindow("deneme",WINDOW_AUTOSIZE);	
	imshow("deneme",img);
	g
	waitKey(0);
	
	return 0;
}
*/


// corner harris txt 4096 version calismiyor olabilir

/*
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include<opencv2/opencv.hpp>
#include <iostream> 
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

using namespace cv;
using namespace std;
// Global variables 
Mat src, src_gray, srcMult;
int thresh = 200;
int max_thresh = 255;

char* source_window = "Source image";
char* corners_window = "Corners detected";
/// Function header 
void cornerHarris_demo(int, void*);

// @function main 
int main() {
	/// Load source image and convert it to gray 
	//char*  imName = "./images/deneme.jpg";
	char*  imName = "lena.jpg";
	src = imread(imName, 1);
	cvtColor(src, src_gray, CV_BGR2GRAY);

	Mat fixedMat = 5*Mat::ones(src.size(),CV_32FC1);
	srcMult = Mat::zeros(src.size(), CV_32FC1);
	for (int i =0 ; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			srcMult.at<float>(i, j) = src_gray.at<uchar>(i, j) * 5;
		}
	}
	//multiply(src,fixedMat,srcMult,1,CV_32FC1);
	//cout << srcMult.at<float>(0,0) << endl;
	//cout << (int)src_gray.at<uchar>(0, 0) << endl;
	/// Create a window and a trackbar 
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
	imshow(source_window, src_gray);

	cornerHarris_demo(0, 0);
	waitKey(0);
	return(0);
}

// @function cornerHarris_demo 
void cornerHarris_demo(int, void*) {
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);

	/// Detector parameters 
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	// Detecting corners 
	cornerHarris(srcMult, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
	/// Normalizing 
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners 
	for (int j = 0; j < dst_norm.rows; j++) {
		for (int i = 0; i < dst_norm.cols; i++) {
			if ((int)dst_norm.at<float>(j, i) > thresh) {
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
	/// Showing the result 
	namedWindow(corners_window, CV_WINDOW_AUTOSIZE);
	imshow(corners_window, dst_norm_scaled);

	// ofstream constructoropens file 
	ofstream outClientFile("histDataCornerHarris.txt", ios::out);


	// exit program if ofstream couldnot open file 
	if (!outClientFile){
		cerr << "File could not be opened" << endl;
		exit(1);
	} // end if 

	for (int i = 0; i < dst_norm_scaled.rows;i++){
		for (int j = 0; j < dst_norm_scaled.cols; j++){
			outClientFile << (int)dst_norm_scaled.at<uchar>(i, j) << endl;
		}
	}
	imwrite("./images/edgeDetectedHigherIntensity.jpg",dst_norm_scaled);
	//cout << dst.at<float>(0, 0);//<<endl<< dst_norm.at<float>(0, 0) <<endl<< dst_norm_scaled.at<float>(0, 0);
	//cout<<" "<<dst_norm.at<float>(123,234);
	//cout << " " << dst_norm_scaled.at<uchar>(12, 113);
	//cout << dst;
}
*/


//corner harris original code do not change

/*
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include<opencv2/opencv.hpp>
#include <iostream> 
#include <stdio.h>
#include <stdlib.h>
#include <fstream>

using namespace cv;
using namespace std;
/// Global variables 
Mat src, src_gray; 
int thresh = 200;
int max_thresh = 255;

char* source_window = "Source image";
char* corners_window = "Corners detected";
/// Function header 
void cornerHarris_demo(int, void*);

// @function main 
int main() {
	/// Load source image and convert it to gray 
	char*  imName = "lena.jpg";
	src = imread(imName, 1);
	cvtColor(src, src_gray, CV_BGR2GRAY);


	/// Create a window and a trackbar 
	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
	createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
	imshow(source_window, src);

	cornerHarris_demo(0, 0);
	waitKey(0);
	return(0);
}

// @function cornerHarris_demo 
void cornerHarris_demo(int, void*) {
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);

	/// Detector parameters 
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	// Detecting corners 
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing 
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners 
	for (int j = 0; j < dst_norm.rows; j++) {
		for (int i = 0; i < dst_norm.cols; i++) {
			if ((int)dst_norm.at<float>(j, i) > thresh) {
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
	/// Showing the result 
	// ofstream constructoropens file 
	ofstream outClientFile("histDataCornerHarris0-255.txt", ios::out);


	// exit program if ofstream couldnot open file 
	if (!outClientFile){
		cerr << "File could not be opened" << endl;
		exit(1);
	} // end if 

	for (int i = 0; i < dst_norm_scaled.rows; i++){
		for (int j = 0; j < dst_norm_scaled.cols; j++){
			outClientFile << (int)dst_norm_scaled.at<uchar>(i, j) << endl;
		}
	}
	namedWindow(corners_window, CV_WINDOW_AUTOSIZE);
	imshow(corners_window, dst_norm_scaled);
	imwrite("./images/edgeDetected.jpg",dst_norm_scaled);
}
*/


/*
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

double getPSNR(const Mat& I1, const Mat& I2);
Scalar getMSSIM(const Mat& I1, const Mat& I2);

static void help()
{
	cout
		<< "------------------------------------------------------------------------------" << endl
		<< "This program shows how to read a video file with OpenCV. In addition, it "
		<< "tests the similarity of two input videos first with PSNR, and for the frames "
		<< "below a PSNR trigger value, also with MSSIM." << endl
		<< "Usage:" << endl
		<< "./video-source referenceVideo useCaseTestVideo PSNR_Trigger_Value Wait_Between_Frames " << endl
		<< "--------------------------------------------------------------------------" << endl
		<< endl;
}

int main()
{
	help();

	//if (argc != 5)
	//{
	//	cout << "Not enough parameters" << endl;
	//	return -1;
	//}

	stringstream conv;

	const string sourceReference = "./video/Megamind.avi", sourceCompareWith = "./video/Megamind_bugy.avi";

	int psnrTriggerValue, delay;
	conv << 35 << endl << 10;       // put in the strings
	conv >> psnrTriggerValue >> delay;        // take out the numbers
	//cout << psnrTriggerValue << endl << delay << endl;
	char c;
	int frameNum = -1;          // Frame counter

	VideoCapture captRefrnc(sourceReference), captUndTst(sourceCompareWith);

	if (!captRefrnc.isOpened())
	{
		cout << "Could not open reference " << sourceReference << endl;
		return -1;
	}

	if (!captUndTst.isOpened())
	{
		cout << "Could not open case test " << sourceCompareWith << endl;
		return -1;
	}

	Size refS = Size((int)captRefrnc.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)captRefrnc.get(CV_CAP_PROP_FRAME_HEIGHT)),
		uTSi = Size((int)captUndTst.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)captUndTst.get(CV_CAP_PROP_FRAME_HEIGHT));

	if (refS != uTSi)
	{
		cout << "Inputs have different size!!! Closing." << endl;
		return -1;
	}

	const char* WIN_UT = "Under Test";
	const char* WIN_RF = "Reference";

	// Windows
	namedWindow(WIN_RF, CV_WINDOW_AUTOSIZE);
	namedWindow(WIN_UT, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_RF, 400, 0);         //750,  2 (bernat =0)
	cvMoveWindow(WIN_UT, refS.width, 0);         //1500, 2

	cout << "Reference frame resolution: Width=" << refS.width << "  Height=" << refS.height
		<< " of nr#: " << captRefrnc.get(CV_CAP_PROP_FRAME_COUNT) << endl;

	cout << "PSNR trigger value " << setiosflags(ios::fixed) << setprecision(3)
		<< psnrTriggerValue << endl;

	Mat frameReference, frameUnderTest;
	double psnrV;
	Scalar mssimV;

	for (;;) //Show the image captured in the window and repeat
	{
		captRefrnc >> frameReference;
		captUndTst >> frameUnderTest;

		if (frameReference.empty() || frameUnderTest.empty())
		{
			cout << " < < <  Game over!  > > > ";
			break;
		}

		++frameNum;
		cout << "Frame: " << frameNum << "# ";

		///////////////////////////////// PSNR ////////////////////////////////////////////////////
		psnrV = getPSNR(frameReference, frameUnderTest);
		cout << setiosflags(ios::fixed) << setprecision(3) << psnrV << "dB";

		//////////////////////////////////// MSSIM /////////////////////////////////////////////////
		if (psnrV < psnrTriggerValue && psnrV)
		{
			mssimV = getMSSIM(frameReference, frameUnderTest);

			cout << " MSSIM: "
				<< " R " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[2] * 100 << "%"
				<< " G " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[1] * 100 << "%"
				<< " B " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[0] * 100 << "%";
		}

		cout << endl;

		////////////////////////////////// Show Image /////////////////////////////////////////////
		imshow(WIN_RF, frameReference);
		imshow(WIN_UT, frameUnderTest);

		c = (char)cvWaitKey(delay);
		if (c == 27) break;
	}

	return 0;
}

double getPSNR(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);        // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // for small values return zero
		return 0;
	else
	{
		double mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0 * log10((255 * 255) / mse);
		return psnr;
	}
}

Scalar getMSSIM(const Mat& i1, const Mat& i2)
{
	const double C1 = 6.5025, C2 = 58.5225;
	//INITS
	int d = CV_32F;

	Mat I1, I2;
	i1.convertTo(I1, d);            // cannot calculate on one byte large values
	i2.convertTo(I2, d);

	Mat I2_2 = I2.mul(I2);        // I2^2
	Mat I1_2 = I1.mul(I1);        // I1^2
	Mat I1_I2 = I1.mul(I2);        // I1 * I2

	// END INITS 

	Mat mu1, mu2;                   // PRELIMINARY COMPUTING
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);

	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);

	Mat sigma1_2, sigma2_2, sigma12;

	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;

	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;

	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	///////////////////////////////// FORMULA ////////////////////////////////
	Mat t1, t2, t3;

	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

	Mat ssim_map;
	divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;

	Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
	return mssim;
}
*/


//video corner detection

/*
#include <iostream> // for standard I/O
#include <string>   // for strings
#include <iomanip>  // for controlling float print precision
#include <sstream>  // string to number conversion

#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

void cornerHarris_demo(int, void*,Mat);

int main(){
	char* videoName = "./video/Megamind.avi";
	int frameNum = -1;
	char c;
	int delay = 10;
	VideoCapture megaMindRef(videoName);
	if (!megaMindRef.isOpened())
	{
		cout << "Could not open reference " << videoName << endl;
		return -1;
	}
	Size refS = Size((int)megaMindRef.get(CV_CAP_PROP_FRAME_WIDTH),(int)megaMindRef.get(CV_CAP_PROP_FRAME_HEIGHT));
	const char* WIN_RF = "Reference";

	// Windows
	namedWindow(WIN_RF, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_RF, 400, 0);

	cout << "Reference frame resolution: Width=" << refS.width << "  Height=" << refS.height
		<< " of nr#: " << megaMindRef.get(CV_CAP_PROP_FRAME_COUNT) << endl;

	Mat frameReference;

	for (;;) //Show the image captured in the window and repeat
	{
		megaMindRef >> frameReference;
		if (frameReference.empty())
		{
			cout << " < < <  Game over!  > > > ";
			break;
		}

		++frameNum;
		cout << "Frame: " << frameNum << "# "<<endl;

		imshow(WIN_RF, frameReference);
		cornerHarris_demo(0,0,frameReference);
		c = (char)cvWaitKey(delay);
		cout << c;
		if (c == 27) break;
	}

	return 0;
}

void cornerHarris_demo(int, void*,Mat src) {
	Mat src_gray,dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);
	int thresh = 200;
	char* corners_window = "Video";
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Detector parameters 
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	// Detecting corners 
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing 
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners 
	for (int j = 0; j < dst_norm.rows; j++) {
		for (int i = 0; i < dst_norm.cols; i++) {
			if ((int)dst_norm.at<float>(j, i) > thresh) {
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
	/// Showing the result 
	namedWindow(corners_window, CV_WINDOW_AUTOSIZE);
	imshow(corners_window, dst_norm_scaled);
}
*/


//gaussian Filter

/*
#include<iostream>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;
Mat takeGaussian(const char* imName,double sigma,int matrixSize){
	Mat A = imread(imName,CV_LOAD_IMAGE_GRAYSCALE);
	double us;
	double squareSigma = sigma*sigma;
	double e = 2.718;
	double newValOfPixel = 0;
	double weights = 0;
	for (int i = matrixSize; i < A.rows-matrixSize; i++){
		for (int k = matrixSize; k < A.cols - matrixSize; k++){
			newValOfPixel = 0;
			weights = 0;
			for (int j = -matrixSize/2; j < matrixSize-1; j++){
				for (int m = -matrixSize/2; m < matrixSize-1; m++){
					us = -pow((j*j + m*m), 1) / (2 * squareSigma);
					double curWeight = pow(e, us);
					newValOfPixel += A.at<uchar>(i+j, k+m)*pow(e, us);
					weights += curWeight;
				}
			}
			newValOfPixel /= matrixSize*matrixSize;
			//cout << round(newValOfPixel) << " "<<(int)A.at<uchar>(i, k)<<" "<<weights<<endl;
			if (newValOfPixel){
				A.at<uchar>(i, k) = round(newValOfPixel);
			}
		}
	}
	return A;
}
int main(){
	const char* imName = "lena.jpg";
	double sigma = 1.0;
	int matrixSize = 5;
	//Mat originalImage = imread(imName,CV_LOAD_IMAGE_GRAYSCALE);
	//namedWindow("original Image",WINDOW_AUTOSIZE);
	//imshow("original Image",originalImage);
	Mat gaussFilteredImage1 = takeGaussian(imName, sigma, matrixSize);
	namedWindow("filtered vs-1", WINDOW_AUTOSIZE);
	imshow("filtered vs-1", gaussFilteredImage1);

	Mat gaussFilteredImage2 = takeGaussian(imName, sigma, 3);
	namedWindow("filtered vs-2", WINDOW_AUTOSIZE);
	imshow("filtered vs-2", gaussFilteredImage2);

	waitKey(0);
	return 0;
}
*/


/*
#include<iostream>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

Mat applyKernelOrtalama(Mat kernel,Mat img){
	if (kernel.rows!=kernel.cols){
		exit(0);
	}
	int starting = kernel.rows / 2;
	int sum;
	int cnt;
	Mat kernelApplied=Mat_<uchar>(img.rows,img.cols*img.channels(),CV_8UC3);
	for (int i = starting; i < img.rows - starting; i++){
		for (int k = starting; k < img.cols*img.channels() - starting; k++){
			//img.at<uchar>(i, k) = calcKernel();
			sum = 0;
			cnt = 0;
			for (int j = -starting; j <= starting; j++){
				for (int m = -starting; m <= starting; m++){
					sum += img.at<uchar>(i, k)*kernel.at<uchar>(starting+j,starting+m);
					cnt++;
				}
			}
			kernelApplied.at<uchar>(i, k) = sum/ kernel.rows*kernel.rows;
		}
	}
	return kernelApplied;
}

Mat applyKernelOrtalamaArray(int *kernel, Mat img){
	int starting = 3 / 2;
	int sum;
	int cnt;
	Mat kernelApplied = Mat_<uchar>(img.rows, img.cols*img.channels(), CV_8UC3);

	for (int i = starting; i < img.rows - starting; i++){
		for (int k = starting; k < img.cols*img.channels() - starting; k++){
			//img.at<uchar>(i, k) = calcKernel();
			sum = 0;
			cnt = 0;
			for (int j = -starting; j <= starting; j++){
				for (int m = -starting; m <= starting; m++){
					sum += img.at<uchar>(i, k)*kernel[cnt];
					cnt++;
				}
			}
			kernelApplied.at<uchar>(i, k) = sum / 9;
		}
	}
	return kernelApplied;
}

int main(){
	const char* imName = "lena.jpg";
	int matrixSize = 3;
	Mat originalImage = imread(imName,CV_LOAD_IMAGE_GRAYSCALE);
	//namedWindow("original Image",WINDOW_AUTOSIZE);
	//imshow("original Image",originalImage);
	Mat kernel1 = Mat::ones(matrixSize,matrixSize,CV_8UC3);

	Mat kernel2 = Mat::zeros(matrixSize, matrixSize, CV_8UC3);
	kernel2.at<uchar>(1, 1) = 1;

	int arr[9];
	for (int i = 0; i < 9; i++){
		arr[i] = 1;
	}
	//Mat kern = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	//Mat kern = (Mat_<float>(3, 3) << 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9);
		int kernel_size = 3;
		Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);
		/// Apply filter filter2D(src, dst, ddepth , kernel, anchor, delta, BORDER_DEFAULT ); 

		Mat K = Mat_<float>(originalImage.rows, originalImage.cols*originalImage.channels(), CV_32F);
		filter2D(originalImage, K, originalImage.depth(), kernel);
	Mat ortalamaImage1 = applyKernelOrtalama(kernel1, originalImage);
	namedWindow("filtered vs-1", WINDOW_AUTOSIZE);
	imshow("filtered vs-1", K);

	Mat ortalamaImage2 = applyKernelOrtalamaArray(arr, originalImage);
	namedWindow("filtered vs-2", WINDOW_AUTOSIZE);
	imshow("filtered vs-2", ortalamaImage2);

	namedWindow("originalImage vs-2", WINDOW_AUTOSIZE);
	imshow("originalImage vs-2", originalImage);
	waitKey(0);
	return 0;
}
*/


//Smoothing an image with 3 by 3 kernel matrix

/*
#include<iostream>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

int main(){
	int kernelSize = 3;
	Mat kernel = Mat::ones(kernelSize,kernelSize,CV_32F)/(float)(kernelSize*kernelSize);
	Mat dst;
	char* imName = "lena.jpg";
	Mat img = imread(imName,CV_LOAD_IMAGE_COLOR);
	filter2D(img, dst, img.depth(), kernel);
	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", img);

	namedWindow("Filtered Image",WINDOW_AUTOSIZE);
	imshow("Filtered Image",dst);
	waitKey(0);

	return 0;
}
*/


//Smoothing an image using 5 by 5 matrix

/*
#include<iostream>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

int main(){
	int kernelSize = 5;
	Mat kernel = Mat::ones(kernelSize, kernelSize, CV_32F) / (float)(kernelSize*kernelSize);
	Mat dst;
	char* imName = "lena.jpg";
	Mat img = imread(imName, CV_LOAD_IMAGE_COLOR);
	filter2D(img, dst, img.depth(), kernel);
	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", img);

	namedWindow("Filtered Image", WINDOW_AUTOSIZE);
	imshow("Filtered Image", dst);
	waitKey(0);

	return 0;
}
*/


//Smoothing an image extreme version

/*
#include<iostream>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

int main(){
	int kernelSize = 35;
	Mat kernel = Mat::ones(kernelSize, kernelSize, CV_32F) / (float)(kernelSize*kernelSize);
	Mat dst;
	char* imName = "lena.jpg";
	Mat img = imread(imName, CV_LOAD_IMAGE_COLOR);
	filter2D(img, dst, img.depth(), kernel);
	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", img);

	namedWindow("Filtered Image", WINDOW_AUTOSIZE);
	imshow("Filtered Image", dst);
	waitKey(0);

	return 0;
}
*/


//image contrast enhancement method with 3 by 3 matrix

/*
#include<iostream>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

int main(){
	int kernelSize = 3;
	Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0,
									-1, 5, -1,
									0, -1, 0);
	Mat dst;
	char* imName = "lena.jpg";
	Mat img = imread(imName, CV_LOAD_IMAGE_COLOR);
	filter2D(img, dst, img.depth(), kernel);
	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", img);

	namedWindow("Filtered Image", WINDOW_AUTOSIZE);
	imshow("Filtered Image", dst);
	waitKey(0);

	return 0;
}
*/


// apply image gaussian method to blur the image with 3 by 3 matrix

/*
#include<iostream>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

int main(){
	char* imName = "lena.jpg";
	Mat img = imread(imName, CV_LOAD_IMAGE_COLOR);

	Mat dst;
	GaussianBlur(img,dst,Size(3,3),0.5,0.5);
	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", img);

	namedWindow("Gaussian Blurred Image", WINDOW_AUTOSIZE);
	imshow("Gaussian Blurred Image", dst);
	waitKey(0);

	return 0;
}
*/


// apply image gaussian method to blur the image with 5 by 5 matrix and with the sigma values 0.5

/*
#include<iostream>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

int main(){
	char* imName = "lena.jpg";
	Mat img = imread(imName, CV_LOAD_IMAGE_COLOR);

	Mat dst;
	GaussianBlur(img, dst, Size(5, 5), 0.5, 0.5);
	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", img);

	namedWindow("Gaussian Blurred Image", WINDOW_AUTOSIZE);
	imshow("Gaussian Blurred Image", dst);
	waitKey(0);

	return 0;
}
*/


// apply image gaussian method to blur the image with 5 by 5 matrix and with the sigma values 1.5

/*
#include<iostream>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

int main(){
	char* imName = "lena.jpg";
	Mat img = imread(imName, CV_LOAD_IMAGE_COLOR);

	Mat dst;
	GaussianBlur(img, dst, Size(5, 5), 1.5, 1.5);
	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", img);

	namedWindow("Gaussian Blurred Image", WINDOW_AUTOSIZE);
	imshow("Gaussian Blurred Image", dst);
	waitKey(0);

	return 0;
}
*/


// calculate derivative of an image using sobel method

/*
#include<iostream>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

#include<fstream>
#include <iomanip> 
#include <string> 
#include <cstdlib>

using namespace std;
using namespace cv;

int main(){
	int scale = 1; 
	int delta = 0; 
	int ddepth = CV_16S;

	//char* imName = "./images/lenaCompressed80Percent.jpg";
	char* imName = "./images/lenaCompressed80Percent.jpg";
	Mat img = imread(imName, CV_LOAD_IMAGE_GRAYSCALE);

	Mat gradX,absGradX,gradY,absGradY,totalGrad,threshImage;
	threshImage = Mat::zeros(img.size(),CV_32F);
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT ); 
	Sobel(img, gradX, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(gradX, absGradX);

	Sobel(img, gradY, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(gradY, absGradY);

	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", img);


	//Total Gradient (approximate) 
	addWeighted( absGradX, 0.5, absGradY, 0.5, 0, totalGrad );

	/// Drawing a circle around edges 
	//cout << totalGrad;
	//cout << totalGrad.channels()<<" "<<totalGrad.cols<<" "<<totalGrad.rows;
	for (int j = 0; j < totalGrad.rows; j++) {
		for (int i = 0; i < totalGrad.cols; i++) {
			if ((int)totalGrad.at<uchar>(j, i) > 150) {
				//cout << Point(i, j) << endl;
				threshImage.at<float>(j,i) = 255;
				//circle(totalGrad, Point(i, j), 1, Scalar(255, 0, 0), 2, 8, 0);//
				//circle(totalGrad, Point(i, j), 1, Scalar(255,0,0),1);
				//circle(absGradX, Point(i, j), 3, Scalar(255, 0, 0), 2, 8, 0);
				//circle(absGradY, Point(i, j), 3, Scalar(255, 0, 0), 2, 8, 0);
			}
		}
	}
	namedWindow("Sobel Applied Image Gradient X", WINDOW_AUTOSIZE);
	imshow("Sobel Applied Image Gradient X", absGradX);
	imwrite("./images/gradXSobel.jpg",absGradX);

	namedWindow("Sobel Applied Image Gradient Y", WINDOW_AUTOSIZE);
	imshow("Sobel Applied Image Gradient Y", absGradY);
	imwrite("./images/gradYSobel.jpg", absGradY);

	namedWindow("Sobel Applied Image Total Gradient", WINDOW_AUTOSIZE);
	imshow("Sobel Applied Image Total Gradient", threshImage);
	imwrite("./images/sobelThresh.jpg", threshImage);

	// ofstream constructoropens file 
	ofstream outClientFile( "histData.txt", ios::out );

	
	// exit program if ofstream couldnot open file 
	if (!outClientFile){
		cerr << "File could not be opened" << endl; 
		exit( 1 ); 
	} // end if 
	
	for (int i = 0; i < totalGrad.rows; i++){
		for (int j = 0; j < totalGrad.cols; j++){
			//cout <<(int) totalGrad.at<uchar>(i, j);
			outClientFile << (int) totalGrad.at<uchar>(i, j)<<endl;
		}
	}
	
	waitKey(0);

	return 0;
}
*/


// calculate derivative of an image using scharr method

/*
#include<iostream>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

int main(){
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	char* imName = "./images/lenaCompressed80Percent.jpg";
	Mat img = imread(imName, CV_LOAD_IMAGE_COLOR);

	Mat gradX, absGradX, gradY, absGradY, totalGrad;
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT ); 
	Scharr(img, gradX, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(gradX, absGradX);

	Scharr(img, gradY, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(gradY, absGradY);

	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", img);

	namedWindow("Sobel Applied Image Gradient X", WINDOW_AUTOSIZE);
	imshow("Sobel Applied Image Gradient X", absGradX);

	namedWindow("Sobel Applied Image Gradient Y", WINDOW_AUTOSIZE);
	imshow("Sobel Applied Image Gradient Y", absGradY);

	//Total Gradient (approximate) 
	addWeighted(absGradX, 0.5, absGradY, 0.5, 0, totalGrad);
	namedWindow("Sobel Applied Image Total Gradient", WINDOW_AUTOSIZE);
	imshow("Sobel Applied Image Total Gradient", absGradY);

	waitKey(0);

	return 0;
}
*/


//Edge DEtection using cornerHarris() function

/*
#include<iostream>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;
Mat img, imgGray;
// @function cornerHarris_demo 
void cornerHarris_demo(int, void*) {
	//detecter parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;
	int thresh = 200;

	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(img.size(), CV_32FC1);


	// Detecting corners 
	cornerHarris(imgGray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
	/// Normalizing 
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners 
	for (int j = 0; j < dst_norm.rows; j++) {
		for (int i = 0; i < dst_norm.cols; i++) {
			if ((int)dst_norm.at<float>(j, i) > thresh) {
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0, 255, 0), 2, 8, 0);
			}
		}
	}
	/// Showing the result 
	namedWindow("cornerHarris() Implementation", CV_WINDOW_AUTOSIZE);
	imshow("cornerHarris() Implementation", dst_norm_scaled);
	
}
int main(){
	const char* imName = "lena.jpg";
	Mat dst,dstNormalized,dstNormalizedScaled;
	/// Detector parameters 
	int thresh = 200;

	img = imread(imName,CV_LOAD_IMAGE_COLOR);
	cvtColor(img,imgGray,CV_BGR2GRAY);

	/// Create a window and a trackbar 
	namedWindow("Original Image", CV_WINDOW_AUTOSIZE);
	createTrackbar("Threshold: ", "Original Image", &thresh, 250, cornerHarris_demo);
	imshow("Original Image", img);

	//namedWindow("cornerHarris() Implementation", WINDOW_AUTOSIZE);
	//imshow("cornerHarris() Implementation", dstNormalizedScaled);
	cornerHarris_demo(0,0);
	waitKey(0);
	return 0;
}
*/


// median Blur implementation using a kernel having size 5

/*
#include<iostream>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

using namespace std;
using namespace cv;

int main(){
	const char* imName = "lena.jpg";
	Mat srcImg = imread(imName,CV_LOAD_IMAGE_GRAYSCALE);
	Mat dst;
	namedWindow("Original Image",WINDOW_AUTOSIZE);
	imshow("Original Image",srcImg);

	medianBlur(srcImg,dst,5);
	namedWindow("Median Filtered Image",WINDOW_AUTOSIZE);
	imshow("Median Filtered Image",dst);
	imwrite("./images/medianBlurred.jpg",dst);

	waitKey(0);
	return 0;
}
*/


//brute force feature description

/*
#include <stdio.h> 
#include <iostream> 
#include "opencv2/core/core.hpp" 
#include "opencv2/features2d/features2d.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/nonfree/features2d.hpp"

using namespace cv;

void readme();

// @function main 
int main(int argc, char** argv){
	char * imName1 = "./images/copyLena.jpg";
	char * imName2 = "./images/lenaDar.jpg";

	//if (argc != 3) { 
	//	return -1; 
	//}
	Mat img_1 = imread(imName1, CV_LOAD_IMAGE_GRAYSCALE); 
	Mat img_2 = imread(imName2, CV_LOAD_IMAGE_GRAYSCALE);

	if (!img_1.data || !img_2.data) { 
		return -1; 
	}
	//-- Step 1: Detect the keypoints using SURF Detector 
	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	detector.detect(img_1, keypoints_1); 
	detector.detect(img_2, keypoints_2);

	//-- Step 2: Calculate descriptors (feature vectors) 
	SurfDescriptorExtractor extractor;

	Mat descriptors_1, descriptors_2;
	extractor.compute(img_1, keypoints_1, descriptors_1); 
	extractor.compute(img_2, keypoints_2, descriptors_2);

	//-- Step 3: Matching descriptor vectors with a brute force matcher 
	BFMatcher matcher(NORM_L2); 
	std::vector< DMatch > matches; 
	matcher.match( descriptors_1, descriptors_2, matches );

	//-- Draw matches 
	Mat img_matches; 
	drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );

	//-- Show detected matches 
	imshow("Matches", img_matches );
	waitKey(0);



	return 0;
}
// @function readme  
void readme() {
	std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl; 
}
*/


//canny edge detector

/*
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include <stdlib.h> 
#include <stdio.h>
#include <iostream>

using namespace cv;

/// Global variables

Mat src, src_gray; 
Mat dst, detected_edges;
Mat ir, dst_norm,dst_norm_scaled;
int edgeThresh = 1; 
int lowThreshold; 
int const max_lowThreshold = 100; 
int ratio = 3; 
int kernel_size = 3; 
char* window_name = "Edge Map";

void CannyThreshold(int, void*) { 
	/// Reduce noise with a kernel 3x3 
	blur( src_gray, detected_edges, Size(3,3) );
	ir = Mat::ones(src_gray.size(), CV_32F);
	for (int i = 0; i < detected_edges.rows; i++){
		for (int j = 0; j < detected_edges.cols; j++){
			ir.at<float>(i, j) = (int)detected_edges.at<uchar>(i, j) * 5;
		}
	}

	//ir = 5 * detected_edges;
	//std::cout << ir;

	/// Canny detector 
	Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
	namedWindow("gray", WINDOW_AUTOSIZE);
	imshow("gray", detected_edges);
	imwrite("./images/cannyEdgeGray.jpg", detected_edges);
	//Canny(ir, ir, lowThreshold, lowThreshold*ratio, kernel_size);
	//std::cout << detected_edges;

	/// Using Canny’s output as a mask, we display our result 
	dst = Scalar::all(0);
	//dst = Mat::ones(ir.size(),CV_32F);
	//src.copyTo(dst, ir);

	//normalize(ir, ir, 0, 1, NORM_MINMAX, CV_32F, Mat());
	//normalize(ir, ir, 0, 1, CV_MINMAX);
	//convertScaleAbs(ir, dst_norm_scaled);

	src.copyTo(dst, detected_edges);
	//std::cout << detected_edges;

	imshow(window_name, dst);
	imwrite("./images/cannyEdgeDetector.jpg", dst);
}

int main(int argc, char** argv) {
	std::cin >> lowThreshold;
	char* imName = "./images/copyLena.jpg";
	//std::cin >> lowThreshold;
	/// Load an image 
	src = imread(imName,CV_LOAD_IMAGE_COLOR);
	namedWindow("original",WINDOW_AUTOSIZE);
	imshow("original",src);
	if (!src.data) {
		return -1;
	}
	/// Create a matrix of the same type and size as src (for dst) 
	dst.create(src.size(), src.type());

	/// Convert the image to grayscale 
	cvtColor( src, src_gray, CV_BGR2GRAY );

	/// Create a window 
	namedWindow( window_name, CV_WINDOW_AUTOSIZE );

	/// Create a Trackbar for user to enter threshold 
	createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );

	/// Show the image 
	CannyThreshold(0, 0);

	/// Wait until user exit program by pressing a key 
	waitKey(0);

	return 0;
}
*/

/*
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include <stdlib.h> 
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;


/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;
Mat ir, dst_norm, dst_norm_scaled;
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

void CannyThreshold(int, void*) {
	/// Reduce noise with a kernel 3x3 
	blur(src_gray, detected_edges, Size(3, 3));
	ir = Mat::ones(src_gray.size(), CV_32F);
	for (int i = 0; i < detected_edges.rows; i++){
		for (int j = 0; j < detected_edges.cols; j++){
			ir.at<float>(i, j) = (int)detected_edges.at<uchar>(i, j) * 5;
		}
	}

	//ir = 5 * detected_edges;
	//std::cout << ir;

	/// Canny detector 
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
	namedWindow("gray", WINDOW_AUTOSIZE);
	imshow("gray", detected_edges);
	//imwrite("./images/checkerDetected.jpg", detected_edges);
	//Canny(ir, ir, lowThreshold, lowThreshold*ratio, kernel_size);
	//std::cout << detected_edges;

	/// Using Canny’s output as a mask, we display our result 
	dst = Scalar::all(0);
	//dst = Mat::ones(ir.size(),CV_32F);
	//src.copyTo(dst, ir);

	//normalize(ir, ir, 0, 1, NORM_MINMAX, CV_32F, Mat());
	//normalize(ir, ir, 0, 1, CV_MINMAX);
	//convertScaleAbs(ir, dst_norm_scaled);

	src.copyTo(dst, detected_edges);
	//std::cout << detected_edges;

	imshow(window_name, dst);
	//imwrite("./images/cannyEdgeDetector.jpg", dst);
}

// @function cornerHarris_demo 
void cornerHarris_demo(int, void*) {
	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);

	/// Detector parameters 
	int thresh = 200;
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	// Detecting corners 
	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
	/// Normalizing 
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners 
	for (int j = 0; j < dst_norm.rows; j++) {
		for (int i = 0; i < dst_norm.cols; i++) {
			if ((int)dst_norm.at<float>(j, i) > lowThreshold*ratio) {
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
	/// Showing the result 
	namedWindow("Corner Harris", CV_WINDOW_AUTOSIZE);
	imshow("Corner Harris", dst_norm_scaled);

	//imwrite("./images/edgeDetectedHigherIntensity.jpg", dst_norm_scaled);
	//cout << dst.at<float>(0, 0);//<<endl<< dst_norm.at<float>(0, 0) <<endl<< dst_norm_scaled.at<float>(0, 0);
	//cout<<" "<<dst_norm.at<float>(123,234);
	//cout << " " << dst_norm_scaled.at<uchar>(12, 113);
	//cout << dst;
}

int main(int argc, char** argv) {
	std::cin >> lowThreshold;
	char* imName = "./images/coin2.jpg";
	//std::cin >> lowThreshold;
	/// Load an image 
	src = imread(imName, CV_LOAD_IMAGE_COLOR);
	namedWindow("original", WINDOW_AUTOSIZE);
	imshow("original", src);
	if (!src.data) {
		return -1;
	}
	/// Create a matrix of the same type and size as src (for dst) 
	dst.create(src.size(), src.type());

	/// Convert the image to grayscale 
	cvtColor(src, src_gray, CV_BGR2GRAY);

	/// Create a window 
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Create a Trackbar for user to enter threshold 
	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
	//createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, cornerHarris_demo);

	/// Show the image 
	CannyThreshold(0, 0);
	//cornerHarris_demo(0, 0);

	/// Wait until user exit program by pressing a key 
	waitKey(0);

	return 0;
}
*/


//Shi-tomasi corner detector goodFeaturesToTrack

/*
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

//Global variables
Mat src, src_gray;

int maxCorners = 23;
int maxTrackbar = 100;
RNG rng(12345);
char * sourceWindow = "Image";

//Function Header
void goodFeaturesToTrackDemo(int, void*);

///main Function
int main(int argc, char** argv) { 
	/// Load source image and convert it to gray 
	char *imName = "./images/copyLena.jpg";
	src = imread( imName, 1 ); 
	cvtColor( src, src_gray, CV_BGR2GRAY );

	/// Create Window 
	namedWindow( sourceWindow, CV_WINDOW_AUTOSIZE );

	/// Create Trackbar to set the number of corners 
	createTrackbar( "Max corners:", sourceWindow, &maxCorners, maxTrackbar, goodFeaturesToTrackDemo );
	imshow(sourceWindow, src);

	goodFeaturesToTrackDemo(0, 0);

	waitKey(0); 
	return(0);
}

//Shi-tomasi corner detector
void goodFeaturesToTrackDemo(int, void*) {
	if (maxCorners < 1) { 
		maxCorners = 1; 
	}
	/// Parameters for Shi-Tomasi algorithm 
	vector<Point2f> corners; 
	double qualityLevel = 0.01; 
	double minDistance = 10; 
	int blockSize = 3; 
	bool useHarrisDetector = false; 
	double k = 0.04;

	/// Copy the source image 
	Mat copy; 
	copy = src.clone();

	/// Apply corner detection 
	goodFeaturesToTrack( src_gray, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k );

	/// Draw corners detected 
	cout<<"** Number of corners detected: "<<corners.size()<<endl; 
	int r = 4; 
	for( int i = 0; i < corners.size(); i++ )
	{
		circle(copy, corners[i], r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
	}
	/// Show what you got 
	namedWindow( sourceWindow, CV_WINDOW_AUTOSIZE ); 
	imshow( sourceWindow, copy );
}
*/


//Feature detection using SurfFeatureDetector

/*
#include <stdio.h> 
#include <iostream> 
#include "opencv2/core/core.hpp" 
#include "opencv2/features2d/features2d.hpp" 
#include "opencv2/nonfree/features2d.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;

void readme();

//main function
int main(){
	char* imName1 = "./images/copyLena.jpg";
	char* imName2 = "./images/lenaCompressed80Percent.jpg";

	Mat img1 = imread(imName1,CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread(imName2, CV_LOAD_IMAGE_GRAYSCALE);

	if (!img1.data || !img2.data){
		std::cout << "Error reading images" << std::endl;
		return -1;
	}

	//Detect the key pint using SURF Detector
	int minHessian = 400;

	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keyPoint1, keyPoint2;

	detector.detect(img1, keyPoint1);
	detector.detect(img2, keyPoint2);

	//Draw keyPoints
	Mat imgKeyPoint1; 
	Mat imgKeyPoint2;

	drawKeypoints(img1,keyPoint1,imgKeyPoint1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
	drawKeypoints(img2, keyPoint2, imgKeyPoint2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	//show detected (drawn) key points
	imshow("KeyPoints 1",imgKeyPoint1);
	imwrite("./images/featureDetectedImage1.jpg", imgKeyPoint1);
	imshow("KeyPoints 2", imgKeyPoint2);
	imwrite("./images/featureDetectedImage2.jpg", imgKeyPoint1);

	waitKey(0);
	return 0;
}
void readme() { 
	std::cout << " Usage: ./SURF_detector <img1> <img2>" << std::endl; 
}
*/


//Feature Matching With Flann

/*
#include <stdio.h> 
#include <iostream> 
#include "opencv2/core/core.hpp" 
#include "opencv2/features2d/features2d.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/nonfree/features2d.hpp"
//#include "opencv2/video/tracking.hpp"

using namespace cv;
using namespace std;
void readme();

int main(int argc, char** argv) {
	char* imName1 = "./images/copyLena.jpg";
	char* imName2 = "./images/lenaDar.jpg";
	//if (argc != 3) { 
	//	readme(); 
	//	return -1; 
	//}
	Mat img1 = imread(imName1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread(imName2, CV_LOAD_IMAGE_GRAYSCALE);
	if (!img1.data || !img2.data) {
		std::cout << " --(!) Error reading images " << std::endl;
		return -1;
	}
	//Detect the key pint using SURF Detector
	int minHessian = 400;

	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keyPoints1, keyPoints2;

	detector.detect(img1, keyPoints1);
	detector.detect(img2, keyPoints2);
	
	cout << keyPoints1.size() << " " << keyPoints1.data() << " " << keyPoints2.size() << keyPoints2.data()<< endl;

	//for (int i = 0; i < keyPoints1.size(); i++){
	//	std::cout << keyPoints1[i]<<" ";
	//}
	//-- Step 2: Calculate descriptors (feature vectors) 
	SurfDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(img1, keyPoints1, descriptors1); 
	extractor.compute(img2, keyPoints2, descriptors2);

	//cout << descriptors1.size() << endl << descriptors2.size() << endl;
	//cout << descriptors1.at<float>(0,0) << endl << descriptors2.at<float>(0,0) << endl;

	//-- Step 3: Matching descriptor vectors using FLANN matcher 
	FlannBasedMatcher matcher; 
	std::vector< DMatch > matches; 
	matcher.match( descriptors1, descriptors2, matches );

	cout << matches.size() << endl;
	//cout << matches[0] << endl;
	double maxDist = 0; 
	double minDist = 100;

	//-- Quick calculation of max and min distances between keypoints 
	for( int i = 0; i < descriptors1.rows; i++ ) { 
		double dist = matches[i].distance; 
		if( dist < minDist ) 
			minDist = dist; 
		if( dist > maxDist ) 
			maxDist = dist; 
	}
	printf("-- Max dist : %f \n", maxDist); 
	printf("-- Min dist : %f \n", minDist);
	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist, 
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very 
	//-- small) 
	//-- PS.- radiusMatch can also be used here. 
	std::vector <DMatch> goodMatches;

	for (int i = 0; i < descriptors1.rows; i++) { 
		if (matches[i].distance <= max(2 * minDist, 0.02)) { 
			goodMatches.push_back(matches[i]); 
		} 
	}

	//-- Draw only "good" matches 
	Mat imgMatches; 
	drawMatches( img1, keyPoints1, img2, keyPoints2, goodMatches, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	//-- Show detected matches 
	imshow( "Good Matches", imgMatches );
	imwrite("./images/matchedImages.jpg", imgMatches);
	waitKey(0);
	return 0;
}
*/


//optical Flow original do not change

/*
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

static void help()
{
	// print a welcome message, and the OpenCV version
	cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
		"Using OpenCV version %s\n" << CV_VERSION << "\n"
		<< endl;

	cout << "\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - auto-initialize tracking\n"
		"\tc - delete all the points\n"
		"\tn - switch the \"night\" mode on/off\n"
		"To add/remove a feature point click it\n" << endl;
}

Point2f point;
bool addRemovePt = false;

static void onMouse(int event, int x, int y, int, void* )
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		point = Point2f((float)x, (float)y);
		addRemovePt = true;
	}
}

int main(int argc, char** argv)
{
	char *videoName = "./video/Megamind.avi";
	VideoCapture cap;
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);

	const int MAX_COUNT = 500;
	bool needToInit = false;
	bool nightMode = false;

	cap.open(videoName);
	//if (argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
	//	cap.open(argc == 2 ? argv[1][0] - '0' : 0);
	//else if (argc == 2)
	//	cap.open(argv[1]);

	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}

	help();

	namedWindow("LK Demo", 1);
	setMouseCallback("LK Demo", onMouse, 0);

	Mat gray, prevGray, image;
	vector<Point2f> points[2];

	for (;;)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty())
			break;

		frame.copyTo(image);
		cvtColor(image, gray, CV_BGR2GRAY);

		if (nightMode)
			image = Scalar::all(0);

		if (needToInit)
		{
			// automatic initialization
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			addRemovePt = false;
		}
		else if (!points[0].empty())
		{
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty())
				gray.copyTo(prevGray);
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
				3, termcrit, 0, 0.001);
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				if (addRemovePt)
				{
					if (norm(point - points[1][i]) <= 5)
					{
						addRemovePt = false;
						continue;
					}
				}

				if (!status[i])
					continue;

				points[1][k++] = points[1][i];
				circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
			points[1].resize(k);
		}

		if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
		{
			vector<Point2f> tmp;
			tmp.push_back(point);
			cornerSubPix(gray, tmp, winSize, cvSize(-1, -1), termcrit);
			points[1].push_back(tmp[0]);
			addRemovePt = false;
		}

		needToInit = false;
		imshow("LK Demo", image);

		char c = (char)waitKey(50);
		if (c == 27)
			break;
		switch (c)
		{
		case 'r':
			needToInit = true;
			break;
		case 'c':
			points[1].clear();
			break;
		case 'n':
			nightMode = !nightMode;
			break;
		default:
			;
		}

		std::swap(points[1], points[0]);
		swap(prevGray, gray);
	}

	return 0;
}
*/


//optical Flow to experiment with

/*
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

static void help()
{
	// print a welcome message, and the OpenCV version
	cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
		"Using OpenCV version %s\n" << CV_VERSION << "\n"
		<< endl;

	cout << "\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - auto-initialize tracking\n"
		"\tc - delete all the points\n"
		"\tn - switch the \"night\" mode on/off\n"
		"To add/remove a feature point click it\n" << endl;
}

Point2f point;
bool addRemovePt = false;
										//flags    params
static void onMouse(int event, int x, int y, int , void* )
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		point = Point2f((float)x, (float)y);
		addRemovePt = true;
	}
}

int main(int argc, char** argv)
{
	char *videoName = "./video/planeVideoShort.mp4";
	VideoCapture cap;
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);

	const int MAX_COUNT = 500;
	bool needToInit = false;
	bool nightMode = false;

	cap.open(videoName);
	//if (argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
	//	cap.open(argc == 2 ? argv[1][0] - '0' : 0);
	//else if (argc == 2)
	//	cap.open(argv[1]);

	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}

	help();

	namedWindow("LK Demo", 1);
	setMouseCallback("LK Demo", onMouse, 0);

	Mat gray, prevGray, image;
	vector<Point2f> points[2];

	for (;;)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty())
			break;

		frame.copyTo(image);
		cvtColor(image, gray, CV_BGR2GRAY);

		if (nightMode)
			image = Scalar::all(0);

		if (needToInit)
		{
			// automatic initialization
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			addRemovePt = false;
		}
		else if (!points[0].empty())
		{
			//cout << points[0];
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty())
				gray.copyTo(prevGray);
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
				3, termcrit, 0, 0.001);
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				if (addRemovePt)
				{
					if (norm(point - points[1][i]) <= 5)
					{
						addRemovePt = false;
						continue;
					}
				}

				if (!status[i])
					continue;

				points[1][k++] = points[1][i];
				circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
			points[1].resize(k);
		}

		if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
		{
			vector<Point2f> tmp;
			tmp.push_back(point);
			cornerSubPix(gray, tmp, winSize, cvSize(-1, -1), termcrit);
			points[1].push_back(tmp[0]);
			addRemovePt = false;
		}

		needToInit = false;
		imshow("LK Demo", image);

		char c = (char)waitKey(50);
		if (c == 27)
			break;
		switch (c)
		{
		case 'r':
			needToInit = true;
			break;
		case 'c':
			points[1].clear();
			break;
		case 'n':
			nightMode = !nightMode;
			break;
		default:
			;
		}

		std::swap(points[1], points[0]);
		swap(prevGray, gray);
	}

	return 0;
}
*/


//opticalFlow to experiment with

/*
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <new>
#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

static void help()
{
	// print a welcome message, and the OpenCV version
	cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
		"Using OpenCV version %s\n" << CV_VERSION << "\n"
		<< endl;

	cout << "\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - auto-initialize tracking\n"
		"\tc - delete all the points\n"
		"\tn - switch the \"night\" mode on/off\n"
		"To add/remove a feature point click it\n" << endl;
}

Point2f point;
bool addRemovePt = false;
//flags    params
static void onMouse(int event, int x, int y, int, void*)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		point = Point2f((float)x, (float)y);
		addRemovePt = true;
	}
}

int main(int argc, char** argv)
{
	char *videoName = "./video/planeVideoShort.mp4";
	VideoCapture cap;
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);

	const int MAX_COUNT = 500;
	bool needToInit = false;
	bool nightMode = false;

	cap.open(videoName);
	
	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}

	help();

	namedWindow("LK Demo", 1);
	setMouseCallback("LK Demo", onMouse, 0);

	Mat gray, prevGray, image,infrared,prevInfrared;
	vector<Point2f> points[2],deneme;

	int j = 0;
	int initialGoodFeatures;
	RNG abc(1);
	//for (int m = 0; m < 100; m++){
	//	cout<<" "<<abc.uniform(0, 8);
	//}
	for (;;)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty())
			break;

		frame.copyTo(image);
		cvtColor(image, gray, CV_BGR2GRAY);
		infrared = Mat::zeros(gray.size(),CV_32F);
		for (int i = 0; i < gray.rows;i++){
			for (int j = 0; j < gray.cols; j++){
				infrared.at<float>(i, j) = abc.uniform(0, 8)*(float)gray.at<uchar>(i, j);
			}
		}
		//cout << infrared;
		//if (nightMode)
		//	image = Scalar::all(0);

		if (j==0)
		{
			// automatic initialization
			cout << endl << "calculating good features to track..." << endl;
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			initialGoodFeatures = points[1].size();
			//cout << endl << initialGoodFeatures << endl;
			//goodFeaturesToTrack(infrared, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			//cout <<endl<<"After gray"<<endl<< points[1];

			cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			//cornerSubPix(infrared, points[1], subPixWinSize, Size(-1, -1), termcrit);
			//cout << "After cornerSubPix" << endl << points[1];

			addRemovePt = false;
		}
		else if (!points[0].empty())
		{
			//cout << points[0];
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty()){
				gray.copyTo(prevGray);
				//infrared.copyTo(prevInfrared);
			}
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
				3, termcrit, 0, 0.001);
			//cout << status.size() << " " << err.size()<<endl;
			if (status.size() <= 0.4*initialGoodFeatures){
				cout << endl << "Recalculating good features to track..." << endl;
				goodFeaturesToTrack(gray, deneme, MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
				//cout <<endl<< points[1].size()<<endl;
				//points[1].resize(deneme.size());
				initialGoodFeatures = deneme.size();
				//cout << initialGoodFeatures<<endl;
				vector<Point2f> points[2];
				points[1] = deneme;
				points[0] = deneme;
				//cout << points[1].size()<<endl;
				//points[1] = new(nothrow) float[150];
				//std::swap(points[1],deneme);
				
				//goodFeaturesToTrack(infrared, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
				cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			}
			//cout << endl<<"baslangic "<<(bool)status[0]<< " " <<err[0]<<endl;
			
			//calcOpticalFlowPyrLK(prevInfrared, infrared, points[0], points[1], status, err, winSize,
			//	3, termcrit, 0, 0.001);
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				if (addRemovePt)
				{
					if (norm(point - points[1][i]) <= 5)
					{
						addRemovePt = false;
						continue;
					}
				}

				if (!status[i])
					continue;

				points[1][k++] = points[1][i];
				circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
			points[1].resize(k);
		}

		if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
		{
			//cout << "iceride" << endl;
			vector<Point2f> tmp;
			tmp.push_back(point);
			cornerSubPix(gray, tmp, winSize, cvSize(-1, -1), termcrit);
			points[1].push_back(tmp[0]);
			addRemovePt = false;
		}

		needToInit = false;
		imshow("LK Demo", image);

		char c = (char)waitKey(30);
		if (c == 27)
			break;

		std::swap(points[1], points[0]);
		swap(prevGray, gray);
		//swap(prevInfrared,infrared);
		j++;
	}
	cout << endl << j;
	while (true){

	}
	return 0;
}
*/


///Optical Flow another sample

/*
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <new>
#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	char *videoName = "./video/planeVideoShort.mp4";
	string directory;
	VideoCapture cap;
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);

	const int MAX_COUNT = 500;
	bool needToInit = false;
	bool nightMode = false;

	cap.open(videoName);

	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}



	namedWindow("LK Demo", 1);

	Mat gray, prevGray, image, infrared, prevInfrared;
	vector<Point2f> points[2], deneme;

	int j = 0;
	int initialGoodFeatures;
	RNG abc(1);
	//for (int m = 0; m < 100; m++){
	//	cout<<" "<<abc.uniform(0, 8);
	//}
	for (;;)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty())
			break;

		frame.copyTo(image);
		cvtColor(image, gray, CV_BGR2GRAY);

		if (j == 0)
		{
			// automatic initialization
			cout << endl << "calculating good features to track..." << endl;
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			initialGoodFeatures = points[1].size();
			cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);

		}
		if (j % 7 == 0){
			std::stringstream ss;
			string add = "thFrame.jpg";
			ss << "./images/" + std::to_string(j) + add;
			directory = ss.str();
			cout << directory << endl;
			imwrite(directory, image);
		}
		else if (!points[0].empty())
		{
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty()){
				gray.copyTo(prevGray);

			}
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
				3, termcrit, 0, 0.001);
			if (status.size() <= 0.4*initialGoodFeatures){
				cout << endl << "Recalculating good features to track..." << endl;
				goodFeaturesToTrack(gray, deneme, MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
				//cout <<endl<< points[1].size()<<endl;
				//points[1].resize(deneme.size());
				initialGoodFeatures = deneme.size();
				//cout << initialGoodFeatures<<endl;
				vector<Point2f> points[2];
				points[1] = deneme;
				points[0] = deneme;
				//cout << points[1].size()<<endl;
				//points[1] = new(nothrow) float[150];
				//std::swap(points[1],deneme);

				//goodFeaturesToTrack(infrared, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
				cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			}
			//cout << endl<<"baslangic "<<(bool)status[0]<< " " <<err[0]<<endl;

			//calcOpticalFlowPyrLK(prevInfrared, infrared, points[0], points[1], status, err, winSize,
			//	3, termcrit, 0, 0.001);
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{


				if (!status[i])
					continue;

				points[1][k++] = points[1][i];
				//circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
			points[1].resize(k);
		}


		needToInit = false;
		imshow("LK Demo", image);

		char c = (char)waitKey(30);
		if (c == 27)
			break;

		std::swap(points[1], points[0]);
		swap(prevGray, gray);
		//swap(prevInfrared,infrared);
		j++;
	}
	cout << endl << j;
	while (true){

	}
	return 0;
}
*/

//drawing functions 

/*
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;
int w = 500;
void MyLine(Mat img, Point start,Point end){
	int thickness = 2;
	int lineType = 8;
	line(img, start, end, Scalar(0, 0, 0), thickness, lineType);

}
void MyEllipse(Mat img,double angle){
	int thickness = 2;
	int lineType = 8;
	ellipse(img, Point(w / 2.0, w / 2.0), Size(w / 4.0, w / 16.0), angle, 0, 360, Scalar(255, 0, 0), thickness, lineType);

}
void MyFilledCircle(Mat img,Point center){
	int thickness = -1;
	int lineType = 8;
	circle(img, center, w / 32.0, Scalar(0, 0,255),thickness,lineType);
}

void MyPolygon(Mat img){
	int lineType = 8;
	// Create some points *
	Point rook_points[1][20]; 
	rook_points[0][0] = Point(w / 4.0, 7 * w / 8.0); 
	rook_points[0][1] = Point(3 * w / 4.0, 7 * w / 8.0); 
	rook_points[0][2] = Point(3 * w / 4.0, 13 * w / 16.0); 
	rook_points[0][3] = Point(11 * w / 16.0, 13 * w / 16.0); 
	rook_points[0][4] = Point(19 * w / 32.0, 3 * w / 8.0); 
	rook_points[0][5] = Point(3 * w / 4.0, 3 * w / 8.0); 
	rook_points[0][6] = Point(3 * w / 4.0, w / 8.0); 
	rook_points[0][7] = Point(26 * w / 40.0, w / 8.0); 
	rook_points[0][8] = Point(26 * w / 40.0, w / 4.0); 
	rook_points[0][9] = Point(22 * w / 40.0, w / 4.0); 
	rook_points[0][10] = Point(22 * w / 40.0, w / 8.0); 
	rook_points[0][11] = Point(18 * w / 40.0, w / 8.0); 
	rook_points[0][12] = Point(18 * w / 40.0, w / 4.0); 
	rook_points[0][13] = Point(14 * w / 40.0, w / 4.0); 
	rook_points[0][14] = Point(14 * w / 40.0, w / 8.0); 
	rook_points[0][15] = Point(w / 4.0, w / 8.0); 
	rook_points[0][16] = Point(w / 4.0, 3 * w / 8.0); 
	rook_points[0][17] = Point(13 * w / 32.0, 3 * w / 8.0); 
	rook_points[0][18] = Point(5 * w / 16.0, 13 * w / 16.0); 
	rook_points[0][19] = Point(w / 4.0, 13 * w / 16.0);

	const Point* ppt[1] = { rook_points[0] }; 
	int npt[] = { 20 };
	fillPoly(img, ppt, npt, 1, Scalar(255, 255, 255), lineType);


}

int main(){
	char atomWindow[] = "Drawing 1: Atom";
	char rookWindow[] = "Drawing 2: Rook";

	Mat atomImage = Mat::zeros(w, w, CV_8UC3);
	Mat rookImage = Mat::zeros(w, w, CV_8UC3);

	/// 1.a. Creating ellipses 
	MyEllipse( atomImage, 90 ); 
	MyEllipse(atomImage, 0);
	MyEllipse(atomImage, 45);
	MyEllipse(atomImage, -45);

	/// 1.b. Creating circles 
	MyFilledCircle( atomImage, Point( w/2.0, w/2.0) );

	/// 2.a. Create a convex polygon 
	MyPolygon( rookImage );

	/// 2.b. Creating rectangles 
	rectangle(rookImage, Point(0, 7 * w / 8.0), Point(w, w), Scalar(0, 255, 255), -1, 8);

	/// 2.c. Create a few lines 
	MyLine( rookImage, Point( 0, 15*w/16 ), Point( w, 15*w/16 ) ); 
	MyLine(rookImage, Point(w / 4, 7 * w / 8), Point(w / 4, w));
	MyLine(rookImage, Point(w / 2, 7 * w / 8), Point(w / 2, w));
	MyLine(rookImage, Point(3 * w / 4, 7 * w / 8), Point(3 * w / 4, w));

	imshow(atomWindow, atomImage);
	imshow(rookWindow,rookImage);
	waitKey(0);
	return 0;
}
*/


//goodFeaturesToTrack x5, xrand(0,8),original comparison

/*
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

int main(){
	char* videoName = "./video/planeVideoShort.mp4";
	const int MAX_COUNT = 500;
	VideoCapture cap;

	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);

	cap.open(videoName);
	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}

	namedWindow("LK Demo", 1);
	//setMouseCallback("LK Demo", onMouse, 0);

	Mat gray, prevGray, image, infrared, prevInfrared;
	vector<Point2f> points[2];

	int j = 0;
	RNG abc(12345);
	//cout << abc.uniform(0, 8);
	for (;;)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty())
			break;

		frame.copyTo(image);
		cvtColor(image, gray, CV_BGR2GRAY);
		infrared = Mat::zeros(gray.size(), CV_32F);
		for (int i = 0; i < gray.rows; i++){
			for (int j = 0; j < gray.cols; j++){
				//infrared.at<float>(i, j) = 5*(int)gray.at<uchar>(i, j);
				infrared.at<float>(i, j) = abc.uniform(0, 8)*(int)gray.at<uchar>(i, j);
			}
		}

		if (j == 0)
		{
			// automatic initialization

			//goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			//cout << points[1];

			goodFeaturesToTrack(infrared, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			//cout << endl << "After gray" << endl << points[1];
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				//if (!status[i])
				//	continue;

				points[1][k++] = points[1][i];
				circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
			imwrite("./images/firstFrameIRRand.jpg",image);
			//cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			//cornerSubPix(infrared, points[1], subPixWinSize, Size(-1, -1), termcrit);
			//cout << "After cornerSubPix" << endl << points[1];
		}
		if (j == 45){
			//goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			//cout << points[1];

			goodFeaturesToTrack(infrared, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			//cout << endl << "After gray" << endl << points[1];
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				//if (!status[i])
				//	continue;

				points[1][k++] = points[1][i];
				circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
			imwrite("./images/45thFrameIRRand.jpg", image);
		}
		else if (!points[0].empty())
		{
			//cout << points[0];
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty()){
				gray.copyTo(prevGray);
				//infrared.copyTo(prevInfrared);
			}
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
				3, termcrit, 0, 0.001);
			cout << status.size() <<" " << err.size()<< endl;
			//calcOpticalFlowPyrLK(prevInfrared, infrared, points[0], points[1], status, err, winSize,
			//	3, termcrit, 0, 0.001);
			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{

				if (!status[i])
					continue;

				points[1][k++] = points[1][i];
				circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
			points[1].resize(k);
		}

		imshow("LK Demo", image);

		char c = (char)waitKey(50);
		if (c == 27)
			break;

		std::swap(points[1], points[0]);
		///swap(prevGray, gray);
		swap(prevInfrared, infrared);
		j++;
	}
	cout << endl<<j<<endl;
	while (true){

	}
	return 0;
}
*/


//Drawing some random geometric shapes

/*
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;
int NUMBER = 50;
int DELAY = 20;
int imgWidth = 800;
int imgHeight = 800;
char* windowName = "window 1";


static Scalar randomColor(RNG& rng) { 
	int icolor = (unsigned)rng; 
	return Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255); 
}


int DrawingRandomLines(Mat image, char* windowName,RNG rng){
	int lineType = 8;
	Point pt1, pt2;
	int x1 =0;
	int y1 = 0;
	int x2 = image.rows - 1;
	int y2 = image.cols - 1;
	for (int i = 0; i < NUMBER; i++) {
		pt1.x = rng.uniform(x1, x2);
		pt1.y = rng.uniform(y1, y2);
		pt2.x = rng.uniform(x1, x2);
		pt2.y = rng.uniform(y1, y2);
		line(image, pt1, pt2, randomColor(rng), rng.uniform(1, 10), 8); 
		imshow(windowName, image);
		if (waitKey(DELAY) >= 0) { 
			return -1; 
		}
	} 
	return 0;
}

int DrawingRandomRectangles(Mat image, char* windowName, RNG rng){
	int lineType = 8;
	Point pt1, pt2;
	int x1 = 0;
	int y1 = 0;
	int x2 = image.rows - 1;
	int y2 = image.cols - 1;
	for (int i = 0; i < NUMBER; i++) {
		pt1.x = rng.uniform(x1, x2);
		pt1.y = rng.uniform(y1, y2);
		pt2.x = rng.uniform(x1, x2);
		pt2.y = rng.uniform(y1, y2);
		rectangle(image, pt1, pt2, randomColor(rng), rng.uniform(1, 10), 8);
		imshow(windowName, image);
		if (waitKey(DELAY) >= 0) {
			return -1;
		}
	}
	return 0;
}

int DrawingRandomEllipses(Mat image, char* windowName, RNG rng){
	int lineType = 8;
	Point pt1, pt2;
	int x1 = 0;
	int y1 = 0;
	int x2 = image.rows - 1;
	int y2 = image.cols - 1;
	for (int i = 0; i < NUMBER; i++) {
		pt1.x = rng.uniform(x1, x2);
		pt1.y = rng.uniform(y1, y2);
		pt2.x = rng.uniform(x1, x2);
		pt2.y = rng.uniform(y1, y2);
		ellipse(image, pt1, Size(rng.uniform(x1, x2)/2, rng.uniform(y1, y2)/2), rng.uniform(0,360), 0, 360, randomColor(rng), rng.uniform(1, 10), 8);
		//ellipse(image, Point(w / 2.0, w / 2.0), Size(w / 4.0, w / 16.0), angle, 0, 360, Scalar(255, 0, 0), thickness, lineType);
		imshow(windowName, image);
		if (waitKey(DELAY) >= 0) {
			return -1;
		}
	}
	return 0;
}

int DrawingRandomCircles(Mat image, char* windowName, RNG rng){
	int lineType = 8;
	Point pt1, pt2;
	int x1 = 0;
	int y1 = 0;
	int x2 = image.rows - 1;
	int y2 = image.cols - 1;
	for (int i = 0; i < NUMBER; i++) {
		pt1.x = rng.uniform(x1, x2);
		pt1.y = rng.uniform(y1, y2);
		pt2.x = rng.uniform(x1, x2);
		pt2.y = rng.uniform(y1, y2);
		circle(image, pt1, rng.uniform(x1,x2),randomColor(rng), rng.uniform(-1,10), 8);
		//circle(img, center, w / 32.0, Scalar(0, 0, 255), thickness, lineType);
		imshow(windowName, image);
		if (waitKey(DELAY) >= 0) {
			return -1;
		}
	}
	return 0;
}

//there is problem in function

int DrawingRandomPolygons(Mat image, char* windowName, RNG rng){
	int lineType = 8;
	Point pt1, pt2;
	int x1 = 0;
	int y1 = 0;
	int x2 = image.rows - 1;
	int y2 = image.cols - 1;
	for (int i = 0; i < NUMBER; i++) {
		pt1.x = rng.uniform(x1, x2);
		pt1.y = rng.uniform(y1, y2);
		pt2.x = rng.uniform(x1, x2);
		pt2.y = rng.uniform(y1, y2);
		//const int size = rng.uniform(x1, 20);
		const int size = 10;
		Point points[1][10];
		for (int j = 0; j < size; j++){
			points[0][i] = Point(rng.uniform(x1, x2), rng.uniform(y1, y2));
		}
		const Point* ppt[1] = { points[0] };
		int npt[] = {10};
		fillPoly(image, ppt, npt,1, randomColor(rng), 8);
		//fillPoly(img, ppt, npt, 1, Scalar(255, 255, 255), lineType);
		imshow(windowName, image);
		if (waitKey(DELAY) >= 0) {
			return -1;
		}
	}
	return 0;
}


int Displaying_Random_Text(Mat image, char* window_name, RNG rng) {
	int lineType = 8;
	int x1 = 0;
	int y1 = 0;
	int x2 = image.rows - 1;
	int y2 = image.cols - 1;
	for (int i = 1; i < NUMBER; i++) {
		Point org; 
		org.x = rng.uniform(x1, x2);
		org.y = rng.uniform(y1, y2);
		putText(image, "Testing text rendering", org, rng.uniform(0, 8), rng.uniform(0, 100)*0.05 + 0.1, randomColor(rng), rng.uniform(1, 10), lineType);

		imshow(window_name, image); if (waitKey(DELAY) >= 0) { return -1; }
	}
	return 0;
}

int Displaying_Big_End(Mat image, char* window_name, RNG rng) {
	Size textsize = getTextSize("OpenCV forever!", CV_FONT_HERSHEY_COMPLEX, 3, 5, 0); 
	Point org((imgWidth - textsize.width) / 2, (imgHeight - textsize.height) / 2);
	int lineType = 8;
	Mat image2;
	for (int i = 0; i < 255; i += 2) {
		image2 = image - Scalar::all(i); 
		putText(image2, "OpenCV forever!", org, CV_FONT_HERSHEY_COMPLEX, 3, Scalar(i, i, 255), 5, lineType);
		imshow(window_name, image2); if (waitKey(DELAY) >= 0) { return -1; }
	}
	return 0;
}


int main(){
	RNG rng(0xFFFFFFFF);
	//initialize a matrix with zeros
	Mat image = Mat::zeros(imgWidth, imgHeight,CV_8UC3);
	imshow(windowName, image);
	int c;

	/// Now, let’s draw some lines 
	c = DrawingRandomLines(image, windowName, rng); 
	if( c != 0 ) 
		return 0;
	
	/// Go on drawing, this time nice rectangles 
	c = DrawingRandomRectangles(image, windowName, rng); 
	if( c != 0 ) 
		return 0;
	
	/// Draw some ellipses 
	c = DrawingRandomEllipses( image, windowName, rng ); 
	if( c != 0 ) 
		return 0;
	
	/// Now some polylines 
	//c = DrawingRandomPolylines( image, windowName, rng ); 
	//if( c != 0 ) 
	//	return 0;
	
	/// Draw filled polygons 
	//c = DrawingRandomPolygons( image, windowName, rng ); 
	//if( c != 0 ) 
	//	return 0;
	

	/// Draw circles 
	c = DrawingRandomCircles( image, windowName, rng ); 
	if( c != 0 ) 
		return 0;
	
	/// Display text in random positions 
	c = Displaying_Random_Text( image, windowName, rng ); 
	if( c != 0 )
		return 0;

	/// Displaying the big end! 
	c = Displaying_Big_End( image, windowName, rng );


	return 0;
}
*/


//Scanning Images

/*
#include<iostream>
#include"opencv2/opencv.hpp"

using namespace std;
using namespace cv;

Mat& ScanImageAndReduceC(Mat& I, const uchar* const table) { 
	// accept only char type matrices 
	CV_Assert(I.depth() != sizeof(uchar));
	int channels = I.channels();
	int nRows = I.rows; int nCols = I.cols * channels;
	if (I.isContinuous()) { 
		nCols *= nRows; nRows = 1; 
	}
	int i, j; uchar* p; 
	for (i = 0; i < nRows; ++i) { 
		p = I.ptr<uchar>(i); 
		for (j = 0; j < nCols; ++j) { 
			p[j] = table[p[j]]; 
		} 
	} 
	return I;
}

Mat& ScanImageAndReduceIterator(Mat& I, const uchar* const table) { 
	// accept only char type matrices 
	CV_Assert(I.depth() != sizeof(uchar));
	const int channels = I.channels(); 
	switch (channels) { 
	case 1: { 
		MatIterator_<uchar> it, end; 
		for (it = I.begin<uchar>(), end = I.end<uchar>(); it != end; ++it) 
			*it = table[*it]; 
		break; 
	} 
	case 3: { 
		MatIterator_<Vec3b> it, end; 
		for (it = I.begin<Vec3b>(), end = I.end<Vec3b>(); it != end; ++it) { 
			(*it)[0] = table[(*it)[0]]; 
			(*it)[1] = table[(*it)[1]]; 
			(*it)[2] = table[(*it)[2]]; 
		} 
	} 
	}
	return I;
}

Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar* const table) { 
	// accept only char type matrices 
	CV_Assert(I.depth() != sizeof(uchar));
	const int channels = I.channels(); 
	switch (channels) {
	case 1:{
		for (int i = 0; i < I.rows; ++i) 
			for (int j = 0; j < I.cols; ++j) 
				I.at<uchar>(i, j) = table[I.at<uchar>(i, j)]; 
		break;
	} 
	case 3: { 
		Mat_<Vec3b> _I = I;
	for (int i = 0; i < I.rows; ++i) for (int j = 0; j < I.cols; ++j) {
		_I(i, j)[0] = table[_I(i, j)[0]]; 
		_I(i, j)[1] = table[_I(i, j)[1]]; 
		_I(i, j)[2] = table[_I(i, j)[2]]; 
	} 
	I = _I; 
	break; 
	}
	}
	return I;
}


int main(){
	int divideWith = 0;
	stringstream s;
	s << 10;
	s >> divideWith;
	if (!s || !divideWith){
		cout << "Invalid number entered for dividing. " << endl; 
		return -1;
	}
	uchar table[256]; 
	for (int i = 0; i < 256; ++i) 
		table[i] = (uchar)(divideWith * (i / divideWith));
	double t = (double)getTickCount(); 
	// do something ... 

	t = ((double)getTickCount() - t)/getTickFrequency(); 
	cout << "Times passed in seconds: " << t << endl;


	return 0;
}
*/


//Eroding and Dilating

/*
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/highgui/highgui.hpp" 
//#include "highgui.h" 
#include <stdlib.h> 
#include <stdio.h>

using namespace cv;

/// Global variables 
Mat src, erosion_dst, dilation_dst;
int erosion_elem = 0; 
int erosion_size = 0; 
int dilation_elem = 0; 
int dilation_size = 0; 
int const max_elem = 2; 
int const max_kernel_size = 21;

// Function Headers 
void Erosion(int, void*); 
void Dilation(int, void*);

// @function main 
int main(int argc, char** argv) { 
	char* imName = "./images/copyLena.jpg";
	/// Load an image 
	src = imread( imName );
	if (!src.data) { 
		return -1; 
	}

	/// Create windows 
	namedWindow( "Erosion Demo", CV_WINDOW_AUTOSIZE ); 
	namedWindow( "Dilation Demo", CV_WINDOW_AUTOSIZE ); 
	cvMoveWindow( "Dilation Demo", src.cols, 0 );

	/// Create Erosion Trackbar 
	createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo", &erosion_elem, max_elem, Erosion );
	createTrackbar("Kernel size:\n 2n +1", "Erosion Demo", &erosion_size, max_kernel_size, Erosion);

	/// Create Dilation Trackbar 
	createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo", &dilation_elem, max_elem, Dilation );
	createTrackbar("Kernel size:\n 2n +1", "Dilation Demo", &dilation_size, max_kernel_size, Dilation);

	/// Default start 
	Erosion( 0, 0 ); 
	Dilation( 0, 0 );

	waitKey(0); 

	return 0;
}

// @function Erosion 
void Erosion(int, void*) {
	int erosion_type; 
	if (erosion_elem == 0){ 
		erosion_type = MORPH_RECT; 
	}
	else if (erosion_elem == 1){ 
		erosion_type = MORPH_CROSS; 
	}
	else if (erosion_elem == 2) { 
		erosion_type = MORPH_ELLIPSE; 
	}
	Mat element = getStructuringElement(erosion_type, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));

	/// Apply the erosion operation 
	erode( src, erosion_dst, element ); 
	imshow( "Erosion Demo", erosion_dst );
}

// @function Dilation 
void Dilation(int, void*) {
	int dilation_type; 
	if (dilation_elem == 0){ 
		dilation_type = MORPH_RECT; 
	}
	else if (dilation_elem == 1){ 
		dilation_type = MORPH_CROSS; 
	}
	else if (dilation_elem == 2) { 
		dilation_type = MORPH_ELLIPSE; 
	}

	Mat element = getStructuringElement(dilation_type, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size)); 

	/// Apply the dilation operation 
	dilate( src, dilation_dst, element ); 
	imshow( "Dilation Demo", dilation_dst );
}
*/


//Adding border to the images

/*
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/highgui/highgui.hpp" 
#include <stdlib.h>
#include <stdio.h>

using namespace cv;

/// Global Variables
Mat src, dst; 
int top, bottom, left, right; 
int borderType; 
Scalar value; 
char* window_name = "copyMakeBorder Demo"; 
RNG rng(12345);

// @function main 
int main(int argc, char** argv) {
	int c;
	char* imName = "./images/copyLena.jpg";
	/// Load an image 
	src = imread( imName,CV_LOAD_IMAGE_COLOR);
	if (!src.data) { 
		return -1; 
		printf(" No data entered, please enter the path to an image file \n"); 
	}
	/// Brief how-to for this program 
	printf( "\n \t copyMakeBorder Demo: \n" );
	printf( "\t -------------------- \n" ); 
	printf( " ** Press 'c' to set the border to a random constant value \n"); 
	printf( " ** Press 'r' to set the border to be replicated \n"); 
	printf( " ** Press 'ESC' to exit the program \n");

	/// Create window 
	namedWindow( window_name, CV_WINDOW_AUTOSIZE );

	/// Initialize arguments for the filter 
	top = (int) (0.05*src.rows); 
	bottom = (int) (0.05*src.rows); 
	left = (int) (0.05*src.cols); 
	right = (int) (0.05*src.cols); dst = src;

	imshow(window_name, dst);

	while (true) {
		c = waitKey(500);
		if ((char)c == 27) { 
			break; 
		}
		else if ((char)c == 'c') {
			borderType = BORDER_CONSTANT; 
		}
		else if ((char)c == 'r') { 
			borderType = BORDER_REPLICATE; 
		}
		value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)); 
		copyMakeBorder(src, dst, top, bottom, left, right, borderType, value);

		imshow(window_name, dst);
	}
	return 0;
}
*/

//Hough Line Transform

/*
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv; 
using namespace std;
void help() { 
	cout << "\nThis program demonstrates line finding with the Hough transform.\n" 
		"Usage:\n" 
		"./houghlines <image_name>, Default is pic1.jpg\n" << endl; 
}
int main(int argc, char** argv) {
	const char* filename = argc >= 2 ? argv[1] : "./images/deneme.jpg";

	Mat src = imread(filename, 0); 
	if (src.empty()) { 
		help(); 
		cout << "can not open " << filename << endl; 
		return -1; }
	Mat dst, cdst; 
	Canny(src, dst, 50, 200, 3); 
	cvtColor(dst, cdst, CV_GRAY2BGR);

#if 0 
	vector<Vec2f> lines; 
	HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );
	for (size_t i = 0; i < lines.size(); i++) { 
		float rho = lines[i][0], theta = lines[i][1]; 
		Point pt1, pt2; 
		double a = cos(theta), b = sin(theta); 
		double x0 = a*rho, y0 = b*rho; 
		pt1.x = cvRound(x0 + 1000 * (-b)); 
		pt1.y = cvRound(y0 + 1000 * (a)); 
		pt2.x = cvRound(x0 - 1000 * (-b)); 
		pt2.y = cvRound(y0 - 1000 * (a)); 
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA); 
	} 
#else 
	vector<Vec4i> lines; 
	HoughLinesP(dst, lines, 1, CV_PI / 180, 50, 50, 10); 
	for (size_t i = 0; i < lines.size(); i++) {
		Vec4i l = lines[i]; 
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
	} 
#endif 
	imshow("source", src); 
	imshow("detected lines", cdst);

	waitKey();
	return 0;
}
*/

//Hough Circles

/*
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include <iostream> 
#include <stdio.h>

using namespace cv;

//function main
int main(int argc, char** argv) {
	char* imName = "./images/circlesOvals.png";
	Mat src, src_gray;

	/// Read the image 
	src = imread( imName, 1 );
	if (!src.data) { 
		return -1; 
	}

	/// Convert it to gray 
	cvtColor( src, src_gray, CV_BGR2GRAY );

	/// Reduce the noise so we avoid false circle detection 
	GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

	vector<Vec3f> circles;

	/// Apply the Hough Transform to find the circles 
	HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );

	/// Draw the circles detected 
	for( size_t i = 0; i < circles.size(); i++ ) { 
		Point center(cvRound(circles[i][0]), 
			cvRound(circles[i][1])); 
		int radius = cvRound(circles[i][2]); 

		// circle center 
		circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 ); 
		
		// circle outline 
		circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 ); 
	}

	/// Show your results 
	namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE ); 
	imshow( "Hough Circle Transform Demo", src );

	waitKey(0); 
	return 0;
}
*/


//mapping implementation

/*
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream> 
#include <stdio.h>
using namespace cv;

/// Global variables 
Mat src, dst;
Mat map_x, map_y; 
char* remap_window = "Remap demo"; 
int ind = 0;

/// Function Headers 
void update_map( void );

// @function main  
int main(int argc, char** argv) { 

	char* imName = "./images/copyLena.jpg";
	/// Load the image 
	src = imread( imName, 1 );

	/// Create dst, map_x and map_y with the same size as src: 
	dst.create( src.size(), src.type() ); 
	map_x.create( src.size(), CV_32FC1 ); 
	map_y.create( src.size(), CV_32FC1 );

	/// Create window 
	namedWindow( remap_window, CV_WINDOW_AUTOSIZE );

	/// Loop 
	while( true ) { 
		/// Each 1 sec. Press ESC to exit the program 
		int c = waitKey( 1000 );
		if ((char)c == 27) { 
			break; 
		}

		/// Update map_x & map_y. Then apply remap 
		update_map(); 
		remap( src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0, 0) );

		/// Display results 
		imshow( remap_window, dst );
	} 
	return 0; 
}

void update_map(void) {
	ind = ind % 4;
	for (int j = 0; j < src.rows; j++) {
		for (int i = 0; i < src.cols; i++) {
			switch (ind) {
				case 0: 
					if (i > src.cols*0.25 && i < src.cols*0.75 && j > src.rows*0.25 && j < src.rows*0.75) {
						map_x.at<float>(j, i) = 2 * (i - src.cols*0.25) + 0.5; 
						map_y.at<float>(j, i) = 2 * (j - src.rows*0.25) + 0.5;
					}
					else { 
						map_x.at<float>(j, i) = 0; 
						map_y.at<float>(j, i) = 0; 
					} 
					break; 
				case 1: 
					map_x.at<float>(j, i) = i; 
					map_y.at<float>(j, i) = src.rows - j; 
					break; 
				case 2: 
					map_x.at<float>(j, i) = src.cols - i; 
					map_y.at<float>(j, i) = j; 
					break; 
				case 3: 
					map_x.at<float>(j, i) = src.cols - i; 
					map_y.at<float>(j, i) = src.rows - j; 
					break;
			} 
			// end of switch
		}
	} 
	ind++;
}
*/

//header of the program 

/*
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <cmath>
#include<sstream>

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

//int WINDOWX =442;
//int WINDOWY = 250;
int WINDOWSIZE =15;
float SIGMA = 5;

//Moves input image one pixel right to construct motion
//Assumes input image is CV_8UC1 and returns an image with same 
//size and type(CV_8UC1)
Mat onePixelRight(Mat img);

//Moves input image one pixel left to construct motion
//Assumes input image is CV_8UC1 and returns an image with same 
//size and type(CV_8UC1)
Mat onePixelLeft(Mat img);

//Moves input image one pixel upward to construct motion
//Assumes input image is CV_8UC1 and returns an image with same 
//size and type(CV_8UC1)
Mat onePixelUp(Mat img);

//Moves input image one pixel downward to construct motion
//Assumes input image is CV_8UC1 and returns an image with same 
//size and type(CV_8UC1)
Mat onePixelDown(Mat img);

//Please note that this function includes many different ways to calculate 
//gradient in x direction. One can swtich between different methods by 
//commenting and uncommenting related parts.
//All different implemantations assume input image is type CV_8UC1 and 
//returns images type CV_32F. Current implementation applies the filter
//[1,-8,0,8,-1]/12 to calculate gradient in x direction and after applying 
//mask it smoothes the result with gaussian kernel size 5 and sigma 1
Mat gradientX(Mat img);

//Please note that this function includes many different ways to calculate 
//gradient in y direction. One can swtich between different methods by 
//commenting and uncommenting related parts.
//All different implemantations assume input image is type CV_8UC1 and 
//returns images type CV_32F. Current implementation applies the filter
//[1;-8;0;8;-1]/12 to calculate gradient in x direction and after applying 
//mask it smoothes the result with gaussian kernel size 5 and sigma 1
Mat gradientY(Mat img);

//This function is to better comprehend what img.type() means in opencv
//Function assumes that input is an integer between 0 and 7. Returns 
//a string saying the type of the image. Note this function is taken from
//stackoverflow
string type2str(int type);

//creates a Gaussian Filter size x size and sigma value 'sigma'
//for both x and y directions. The filter returned has type
//CV_32F. The function applies normalization meaning the sum
//of the numbers in the filter is 1
Mat createGaussianFilter(int size, float sigma);

//creates an average filter having dimensions size x size
//The returned filter has type CV_32F. The function applies
// normalization meaning the sum of the number in the filter
// is 1(All numbers in the filter is 1/(size x size))
Mat createAverageFilter(int size);

// Assumes size of the img and size of the kernel are the 
//same. Assumes both of the input images have type CV_32F
//Kernel is the weighting function to be applied to img
//Returning number is a float sum of the correlation of
//the img and kernel
float applyWindowFunction(Mat img, Mat kernel);

//Assumes inputs img and gradient have type CV_32F
//returns an image whose dimensions are 
//kernelSize x kernelSize and also has type CV_32F
//Applies correlation and puts the result corresponding 
//place in the returned image. Please note that img and
//gradient should have the same dimensions with returned
//image (kernelSize x kernelSize)
Mat applyKernel(Mat img, Mat gradient, int kernelSize);

//Assumes img has type CV_8UC1 and row and column are floats
//row representing row number and column representing column number
//calculates the intensity value at that position using
//bilinear interpolation. Calculates result by looking at the
//values at the neighboring pixels( 4 pixel) and returns a float
//number which is intensity value at the subpixel
float bilinearInterpolation(Mat img, float row, float column);

//Assumes both imgFirst and imgSecond have types CV_8UC1 
//column1 and row1 are the upperleft position of the window for
//imgFirst. column2 and row2 are the upperleft position of the
//window for imgSecond. Function takes time derivative between these
//two frames simply subtracks imgFirst  from the imgSecond. Returned
//image has type CV_32F and has dimensions WINDOWSIZE x WINDOWSIZE
//(WINDOWSIZE is a global variable)
Mat createDerivativeTSubPix(Mat imgFirst, Mat imgSecond, float column1, float row1, float column2, float row2);

//I no longer use this function in my code
Mat derivativeT(Mat imgFirst, Mat imgSecond, float dispX, float dispY);

//This is the function version of whole calculations. Assumes 
//imgFirstFrame and  imgSecondFrame have type CV_8UC1 corners
//contains the corner positions, nextCorners contains corner
//positions after calculations function can drop some corners 
//at the nextCorners
int calcOpticalFlowMotion(Mat imgFirstFrame, Mat imgSecondFrame, vector<Point2f> &corners, vector<Point2f> &nextCorners);

// Assumes img has type CV_8UC1 and returns an image
//that has size img.rows/2 x img.cols/2, from img constructs 
//a new image that is one level above the pyramid using Gaussian
//technique. Returned image has type CV_8UC1 also.
Mat constructPyramid(Mat img);


Mat onePixelRight(Mat img){
	//Mat result = Mat::zeros(img.size(), CV_8UC1);
	Mat result;
	result.create(img.size(), CV_8UC1);
	for (int i = 0; i < img.rows; i++){
		for (int j = 1; j < img.cols; j++){
			result.at<uchar>(i, j) = (int)img.at<uchar>(i, j - 1);
			//cout << result.at<float>(i, j) << endl;
		}
	}
	return result;
}

Mat onePixelLeft(Mat img){
	//Mat result = Mat::zeros(img.size(), CV_8UC1);
	Mat result;
	result.create(img.size(), CV_8UC1);
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols-1; j++){
			result.at<uchar>(i, j) = (int)img.at<uchar>(i, j + 1);
			//cout << result.at<float>(i, j) << endl;
		}
	}
	return result;
}

Mat onePixelDown(Mat img){
	//Mat result = Mat::zeros(img.size(), CV_8UC1);
	Mat result;
	result.create(img.size(), CV_8UC1);
	for (int i = 1; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			result.at<uchar>(i, j) = (int)img.at<uchar>(i-1, j);
			//cout << result.at<float>(i, j) << endl;
		}
	}
	return result;
}

Mat onePixelUp(Mat img){
	//Mat result = Mat::zeros(img.size(), CV_8UC1);
	Mat result;
	result.create(img.size(), CV_8UC1);
	for (int i = 0; i < img.rows-1; i++){
		for (int j = 0; j < img.cols; j++){
			result.at<uchar>(i, j) = (int)img.at<uchar>(i + 1, j);
			//cout << result.at<float>(i, j) << endl;
		}
	}
	return result;
}

Mat gradientX(Mat img){
	int scale = 1; 
	int delta = 0; 
	int ddepth = CV_32F;
	Mat gradX = Mat::zeros(img.size(), CV_32F);

	//for (int i = 0; i < img.rows;i++){
	//	for (int j = 0; j < img.cols-1; j++){
	//		gradX.at<float>(i, j) = ((int)img.at<uchar>(i, j+1) - (int)img.at<uchar>(i, j));
	//	}
	//}

	//for (int i = 0; i < img.rows; i++){
	//	for (int j = 1; j < img.cols; j++){
	//		gradX.at<float>(i, j) = ((int)img.at<uchar>(i, j) - (int)img.at<uchar>(i, j-1));
	//	}
	//}

	//for (int i = 0; i < img.rows; i++){
	//	for (int j = 1; j < img.cols-1; j++){
	//		gradX.at<float>(i, j) = ((int)img.at<uchar>(i, j+1) - (int)img.at<uchar>(i, j - 1));
	//	}
	//}
	
	////using prewitt operator
	//float res;
	//for (int i = 1; i < img.rows-1; i++){
	//	for (int j = 1; j < img.cols - 1; j++){
	//		res = 0;
	//		for (int m = -1; m <=1;m++){
	//			res += ((int)img.at<uchar>(i+m, j + 1) - (int)img.at<uchar>(i+m, j - 1));
	//		}
	//		gradX.at<float>(i, j) = res / 3;
	//	}
	//}

	//Sobel(img, gradX, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	//gradX = gradX / 8;

	//for (int i = 0; i < img.rows-1; i++){
	//	for (int j = 0; j < img.cols - 1; j++){
	//		for (int m = 0; m < 2; m++){
	//			gradX.at<float>(i, j) += (int)img.at<uchar>(i + m, j + 1) - (int)img.at<uchar>(i + m, j);
	//		}
	//	}
	//}
	for (int i = 0; i < img.rows; i++){
		for (int j = 2; j < img.cols - 2; j++){
			gradX.at<float>(i, j) += (-1 * (int)img.at<uchar>(i, j - 2) +8* (int)img.at<uchar>(i, j - 1)-8*(int)img.at<uchar>(i, j + 1)+1*(int)img.at<uchar>(i, j +2))/12;
		}
	}
	GaussianBlur(gradX, gradX, Size(5, 5), 1, 1);
	return gradX;
}
Mat gradientY(Mat img){
	int scale = 1;
	int delta = 0;
	int ddepth = CV_32F;
	Mat gradY = Mat::zeros(img.size(),CV_32F);
	//for (int i = 0; i < img.rows - 1; i++){
	//	for (int j = 0; j < img.cols; j++){
	//		gradY.at<float>(i, j) = ((int)img.at<uchar>(i + 1, j) - (int)img.at<uchar>(i, j));
	//	}
	//}

	//for (int i = 1; i < img.rows; i++){
	//	for (int j = 0; j < img.cols; j++){
	//		gradY.at<float>(i, j) = ((int)img.at<uchar>(i, j) - (int)img.at<uchar>(i-1, j));
	//	}
	//}

	//for (int i = 1; i < img.rows-1; i++){
	//	for (int j = 0; j < img.cols; j++){
	//		gradY.at<float>(i, j) = ((int)img.at<uchar>(i+1, j) - (int)img.at<uchar>(i - 1, j));
	//	}
	//}
	
	////using prewitt operator
	//float res;
	//for (int i = 1; i < img.rows - 1; i++){
	//	for (int j = 1; j < img.cols-1; j++){
	//		res = 0;
	//		for (int m = -1; m < 2; m++){
	//			res += ((int)img.at<uchar>(i + 1, j+m) - (int)img.at<uchar>(i - 1, j+m));
	//		}
	//		gradY.at<float>(i, j) = res / 3;
	//	}
	//}

	//Sobel(img, gradY, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	//gradY = gradY / 8;

	//for (int i = 0; i < img.rows-1; i++){
	//	for (int j = 0; j < img.cols-1; j++){
	//		for (int m = 0; m < 2; m++){
	//			gradY.at<float>(i, j) += (int)img.at<uchar>(i+1, j + m) - (int)img.at<uchar>(i, j+m);
	//		}
	//	}
	//}

	for (int i = 2; i < img.rows - 2; i++){
		for (int j = 0; j < img.cols; j++){
			gradY.at<float>(i, j) += (-(int)img.at<uchar>(i - 2, j) + 8 * (int)img.at<uchar>(i - 1, j) -8 * (int)img.at<uchar>(i + 1, j) + (int)img.at<uchar>(i +2, j))/12;
		}
	}
	GaussianBlur(gradY, gradY, Size(5, 5), 1, 1);
	return gradY;
}

string type2str(int type){
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth){
	case CV_8U: r = "8U"; break;
	case CV_8S: r = "8s"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:	r = "User"; break;
	}
	r += "C";
	r += (chans+'0');
	return r;
}

Mat createGaussianFilter(int size,float sigma){
	//Mat result = Mat::zeros(size, size, CV_32F);
	Mat result;
	result.create(size, size, CV_32F);
	float e = 2.718;
	float sum = 0;
	float curResult = 0;
	for (int i = -size / 2; i <= size / 2; i++){
		for (int j = -size / 2; j <= size / 2; j++){
			curResult = pow(e, (-(i*i + j*j) / (2 * sigma*sigma)));
			result.at<float>(i + size / 2, j + size / 2) = curResult;
			sum += curResult;
		}
	}
	//cout << endl<<sum << endl;
	for (int i = -size / 2; i <= size / 2; i++){
		for (int j = -size / 2; j <= size / 2; j++){
			//curResult = pow(e, (-(i*i + j*j) / (2 * sigma*sigma)));
			result.at<float>(i + size / 2, j + size / 2) = result.at<float>(i + size / 2, j + size / 2) / sum;
			//sum += curResult;
		}
	}
	return result;
}

Mat createAverageFilter(int size){
	//Mat result = Mat::zeros(size, size, CV_32F);
	Mat result;
	result.create(size, size, CV_32F);
	
	float sum = size*size;
	for (int i = -size / 2; i <= size / 2; i++){
		for (int j = -size / 2; j <= size / 2; j++){
			result.at<float>(i + size / 2, j + size / 2) = 1/sum;
		}
	}
	return result;
}

//applies window function at the center of an image
float applyWindowFunction(Mat img,Mat kernel){
	float result = 0;
	//int startR = img.rows / 2 - kernel.rows / 2;
	//int startR = 260;
	//int endR = img.rows / 2 + kernel.rows / 2;

	//int startC = img.cols / 2 - kernel.cols / 2;
	//int startC = 360;
	//int endC = img.cols / 2 + kernel.cols / 2;

	for (int i =0; i < kernel.rows; i++){
		for (int j = 0; i < kernel.cols; i++){
			result += img.at<float>(i,j)*kernel.at<float>(i,j);
		}
	}
	return result;
}

Mat applyKernel(Mat img,Mat gradient,int kernelSize){
	//Mat result = Mat::zeros(kernelSize,kernelSize,CV_32F);
	Mat result;
	result.create(kernelSize,kernelSize, CV_32F);
	for (int i = 0; i < kernelSize; i++){
		for (int j = 0; j < kernelSize; j++){
			result.at<float>(i, j) = img.at<float>(i , j )*gradient.at<float>(i , j );
		}
	}
	return result;
}

float bilinearInterpolation(Mat img,float row, float column){
	//Assumes input image is uchar type
	float alfaX = row - floor(row);
	float alfaY = column - floor(column);
	float res = (1 - alfaX)*(1 - alfaY)*(int)img.at<uchar>(floor(row), floor(column));
	res += alfaX*(1 - alfaY)*(int)img.at<uchar>(floor(row) + 1, floor(column));
	res += (1 - alfaX)*alfaY*(int)img.at<uchar>(floor(row), floor(column) + 1);
	res += alfaX*alfaY*(int)img.at<uchar>(floor(row) + 1, floor(column) + 1);
	return res;
}

Mat createDerivativeTSubPix(Mat imgFirst,Mat imgSecond, float column1,float row1,float column2,float row2){
	Mat result = Mat::zeros(WINDOWSIZE,WINDOWSIZE,CV_32F );
	//Mat result;
	//result.create(WINDOWSIZE, WINDOWSIZE, CV_32F);
	//Mat convFirst = Mat::zeros(imgFirst.size(), CV_32F);
	Mat convFirst;
	convFirst.create(imgFirst.size(),CV_32F);
	//Mat convSecond = Mat::zeros(imgSecond.size(), CV_32F);
	Mat convSecond;
	convSecond.create(imgSecond.size(),CV_32F);
	for (int i = 0; i < imgFirst.rows - 1; i++){
		for (int j = 0; j < imgFirst.cols - 1; j++){
			for (int m = 0; m < 2; m++){
				convFirst.at<float>(i, j) += (int)imgFirst.at<uchar>(i + m, j) + (int)imgFirst.at<uchar>(i + m, j + 1);
			}
		}
	}
	for (int i = 0; i < imgSecond.rows - 1; i++){
		for (int j = 0; j < imgSecond.cols - 1; j++){
			for (int m = 0; m < 2; m++){
				convSecond.at<float>(i, j) += (int)imgSecond.at<uchar>(i + m, j) + (int)imgSecond.at<uchar>(i + m, j + 1);
			}
		}
	}
	for (int i = 0; i < WINDOWSIZE; i++){
		for (int j = 0; j < WINDOWSIZE; j++){
			result.at<float>(i, j) = -bilinearInterpolation(imgFirst, row1 + i, column1 + j);
			result.at<float>(i, j)+= (int)imgSecond.at<uchar>((int)floor(row2) + i, (int)floor(column2) + j);
		}
	}
	GaussianBlur(result, result, Size(5, 5), 1, 1);
	return result;
}

Mat derivativeT(Mat imgFirst, Mat imgSecond, float dispX, float dispY){
	//Mat derivative = Mat::zeros(imgFirst.size(), CV_32F);
	//Mat convFirst = Mat::zeros(imgFirst.size(), CV_32F);
	//Mat convSecond = Mat::zeros(imgSecond.size(), CV_32F);
	Mat derivative;
	derivative.create(imgFirst.size(), CV_32F);
	Mat convFirst;
	convFirst.create(imgFirst.size(), CV_32F);
	Mat convSecond;
	convSecond.create(imgSecond.size(), CV_32F);

	for (int i = 0; i < imgFirst.rows - 1; i++){
		for (int j = 0; j < imgFirst.cols - 1; j++){
			for (int m = 0; m < 2; m++){
				convFirst.at<float>(i, j) += (int)imgFirst.at<uchar>(i + m, j) + (int)imgFirst.at<uchar>(i + m, j + 1);
			}
		}
	}

	for (int i = 0; i < imgSecond.rows - 1; i++){
		for (int j = 0; j < imgSecond.cols - 1; j++){
			for (int m = 0; m < 2; m++){
				convSecond.at<float>(i, j) += (int)imgSecond.at<uchar>(i + m, j) + (int)imgSecond.at<uchar>(i + m, j + 1);
			}
		}
	}
	if (dispX >= 0 && dispY >= 0){
		for (int i = 0; i < imgSecond.rows - round(dispX); i++){
			for (int j = 0; j < imgSecond.cols - round(dispY); j++){
				derivative.at<float>(i, j) += convFirst.at<float>(i + round(dispX), j + round(dispY)) - convSecond.at<float>(i, j);
			}
		}
	}
	else if (dispX >= 0 && dispY <= 0){
		for (int i = 0; i < imgSecond.rows - round(dispX); i++){
			for (int j = 0 - round(dispY); j < imgSecond.cols; j++){
				derivative.at<float>(i, j) += convFirst.at<float>(i + round(dispX), j + round(dispY)) - convSecond.at<float>(i, j);
			}
		}
	}
	else if (dispX <= 0 && dispY >= 0){
		for (int i = 0 - round(dispX); i < imgSecond.rows; i++){
			for (int j = 0; j < imgSecond.cols - round(dispY); j++){
				derivative.at<float>(i, j) += convFirst.at<float>(i + round(dispX), j + round(dispY)) - convSecond.at<float>(i, j);
			}
		}
	}
	else {
		for (int i = 0 - round(dispX); i < imgSecond.rows; i++){
			for (int j = 0 - round(dispY); j < imgSecond.cols; j++){
				derivative.at<float>(i, j) += convFirst.at<float>(i + round(dispX), j + round(dispY)) - convSecond.at<float>(i, j);
			}
		}
	}
	return derivative;
}

int calcOpticalFlowMotion(Mat imgFirstFrame, Mat imgSecondFrame, vector<Point2f> &corners, vector<Point2f> &nextCorners){
	Mat gradX, gradY, gradWholeX, gradWholeY, Ixx, Ixy, Iyx, Iyy, gaussFilter, averageFilter;
	Mat diffFrame, diffWholeFrame, R1, R2, copyFirst, imgOriginal, imgColorful, gray;
	size_t j, m;
	int iterNum = 0;
	float G11, G12, G21, G22, res1, res2, det, dispX, dispY;
	dispX = 0;
	dispY = 0;
	float WINDOWX1;
	float WINDOWY1;
	float WINDOWX2;
	float WINDOWY2;

	gaussFilter = createGaussianFilter(WINDOWSIZE, SIGMA);
	gradWholeX = gradientX(imgFirstFrame);
	gradWholeY = gradientY(imgFirstFrame);

	for (int i = 0; i < corners.size(); i++){
		if (corners.at(i).x - WINDOWSIZE / 2 < 0 || corners.at(i).x + WINDOWSIZE / 2 >= imgFirstFrame.cols){
			corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
			i--;
			continue;
		}
		else if (corners.at(i).y - WINDOWSIZE / 2<0 || corners.at(i).y + WINDOWSIZE / 2 >= imgFirstFrame.rows){
			corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
			i--;
			continue;
		}
		else{
			WINDOWX1 = floor(corners.at(i).x - WINDOWSIZE / 2);
			WINDOWY1 = floor(corners.at(i).y - WINDOWSIZE / 2);

			WINDOWX2 = floor(corners.at(i).x - WINDOWSIZE / 2);
			WINDOWY2 = floor(corners.at(i).y - WINDOWSIZE / 2);
		}
		gradX = gradWholeX(Rect(WINDOWX1, WINDOWY1, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
		gradY = gradWholeY(Rect(WINDOWX1, WINDOWY1, WINDOWSIZE, WINDOWSIZE)); // using a rectangle

		Ixx = applyKernel(gradX, gradX, WINDOWSIZE);
		Ixy = applyKernel(gradX, gradY, WINDOWSIZE);
		Iyx = applyKernel(gradY, gradX, WINDOWSIZE);
		Iyy = applyKernel(gradY, gradY, WINDOWSIZE);


		G11 = applyWindowFunction(Ixx, gaussFilter);
		G12 = applyWindowFunction(Ixy, gaussFilter);
		G21 = applyWindowFunction(Iyx, gaussFilter);
		G22 = applyWindowFunction(Iyy, gaussFilter);

		if (imgFirstFrame.size() != imgSecondFrame.size()){
			cerr << "Images have different sizes" << endl;
			return -1;
		}
		dispX = dispY = 0;

	label:
		if (WINDOWX1 + WINDOWSIZE>imgFirstFrame.cols || WINDOWX1<0){
			corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
			i--;
			continue;
		}
		else if (WINDOWY1 + WINDOWSIZE>imgFirstFrame.rows || WINDOWY1<0){
			corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
			i--;
			continue;
		}
		else{
			diffFrame = createDerivativeTSubPix(imgFirstFrame, imgSecondFrame, WINDOWX1, WINDOWY1, WINDOWX2, WINDOWY2);
		}
		R1 = applyKernel(diffFrame, gradX, WINDOWSIZE);
		R2 = applyKernel(diffFrame, gradY, WINDOWSIZE);

		res1 = applyWindowFunction(R1, gaussFilter);
		res2 = applyWindowFunction(R2, gaussFilter);
		det = G22*G11 - G12*G21;
		dispX = (G22 *res1 - G12 *res2) / det;
		dispY = (G11 *res2 - G21 *res1) / det;

		if (isnan(dispX) || isnan(dispY)){
			corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
			i--;
			continue;
		}
		else if (corners.at(i).x + dispX<0 || corners.at(i).x + dispX >= imgFirstFrame.cols){
			corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
			i--;
			continue;
		}
		else if (corners.at(i).y + dispY<0 || corners.at(i).y + dispY >= imgFirstFrame.rows){
			corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
			i--;
			continue;
		}
		else{
			corners.at(i).x += dispX;
			corners.at(i).y += dispY;
			if (abs(dispX) > 0.1 || abs(dispY) > 0.1){
				iterNum++;
				if (iterNum > 10){
					iterNum = 0;
					corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
					i--;
					continue;
				}
				WINDOWX1 -= dispX;
				WINDOWY1 -= dispY;
				goto label;
			}
			iterNum = 0;

		}
	}
	nextCorners = corners;
	cout << "inside the function " << corners.size()<<" "<<nextCorners.size()<<endl;
	return 0;
}

Mat constructPyramid(Mat img){
	Mat resultX = Mat::zeros(img.rows, img.cols / 2,CV_8UC1);
	Mat result = Mat::zeros(img.rows/2,img.cols/2,CV_8UC1);
	//Mat kernel = (Mat_<float>(5, 5) << 1 / 16, 1 / 4, 6 / 16, 4 / 16, 1 / 16, 4 / 16, 16 / 16, 24 / 16, 16 / 16, 4 / 16, 6 / 16, 24 / 16, 36 / 16, 24 / 16, 6 / 16, 4 / 16, 16 / 16, 24 / 16, 16 / 16, 4 / 16, 1 / 16, 1 / 4, 6 / 16, 4 / 16, 1 / 16);
	Mat kernel = createGaussianFilter(5,1);
	Mat kernel1DX = (Mat_<float>(1,5)<<0.05,0.25,0.4,0.25,0.05);
	Mat kernel1DY = (Mat_<float>(5,1)<<0.05,0.25,0.4,0.25,0.05);

	//for (int i = 0; i < result.rows-3;i++){
	//	for (int j = 0; j < result.cols-3; j++){
	//		for (int m = 0; m < 5; m++){
	//			for (int n = 0; n < 5; n++){
	//				result.at<uchar>(i, j) += (int)img.at<uchar>(2 * i + m, 2 * j + n)*kernel.at<float>(m,n);
	//			}
	//		}
	//	}
	//}

	for (int i = 0; i < resultX.rows; i++){
		for (int j = 0; j < resultX.cols - 2; j++){
			for (int m = 0; m < 5; m++){
				resultX.at<uchar>(i, j+2) += (int)img.at<uchar>(i , 2 * j + m)*kernel1DX.at<float>(0, m);		
			}
			//cout << "i, j are " << i << " " << j << endl;
		}
	}
	for (int i = 0; i < result.rows-2; i++){
		for (int j = 0; j < result.cols; j++){
			for (int m = 0; m < 5; m++){
				result.at<uchar>(i+2, j) += (int)resultX.at<uchar>(2 * i + m, j)*kernel1DY.at<float>(m, 0);
			}
			//cout << "i, j are " << i << " " << j << endl;
		}
	}
	
	return result;
}

*/


//iterative version

/*
int main(){
	float WINDOWX1;
	float WINDOWY1;

	float WINDOWX2;
	float WINDOWY2;

	//char* imName = "./images/copyLena.jpg";
	char* imName = "./images/lenaDar.jpg";
	//char* imName = "./images/lenaDar.jpg";
	//char* imName = "./images/0thFrame.jpg";
	//char* imName = "./images/7thFrame.jpg";
	//char* imName = "./images/14thFrame.jpg";
	Mat imgSecondFrame, gradX, gradY, gradWholeX, gradWholeY, Ixx, Ixy, Iyx, Iyy, gaussFilter, averageFilter;
	Mat diffFrame, diffWholeFrame, R1, R2, copyFirst, imgOriginal, imgColorful,subPixelValues;
	float G11, G12, G21, G22, res1, res2, det, dispX, dispY;
	int initialCorner;

	Mat imgFirstFrame = imread(imName, CV_LOAD_IMAGE_GRAYSCALE);
	imgOriginal = imgFirstFrame.clone();
	imgSecondFrame = onePixelRight(imgFirstFrame);
	imgSecondFrame = onePixelRight(imgSecondFrame);
	imgSecondFrame = onePixelDown(imgSecondFrame);
	imgSecondFrame = onePixelDown(imgSecondFrame);
	imwrite("./images/imgFirstFrame.jpg",imgFirstFrame);
	imwrite("./images/imgSecondFrame.jpg",imgSecondFrame);

	
	vector<Point2f> corners;
	int maxCorners = 20;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3;
	int iterNum = 0;
	bool useHarrisDetector = false;
	double k = 0.04;
	goodFeaturesToTrack(imgFirstFrame, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
	initialCorner = corners.size();
	size_t j, m;
	for (j = m = 0; j < corners.size(); j++)
	{

		corners[m++] = corners[j];
		circle(imgOriginal, corners[j], 3, Scalar(255, 255, 255), -1, 8);
	}
	namedWindow("imgFirstFrame", WINDOW_AUTOSIZE);
	imshow("imgFirstFrame", imgOriginal);

	//Mat imgOriginalIR = Mat::zeros(imgOriginal.size(), CV_32F);
	//for (int i = 0; i < imgOriginal.rows;i++){
	//	for (int j = 0; j < imgOriginal.cols;j++){
	//		imgOriginalIR.at<float>(i, j) = 4*(int)imgOriginal.at<uchar>(i, j);
	//	}
	//}
	//cout << "Ir version of image is " << imgOriginalIR << endl;
	////pyrDown works with images CV_32F
	//pyrDown(imgOriginalIR,imgOriginalIR,Size(imgOriginalIR.cols/2,imgOriginalIR.rows/2));
	//namedWindow("Reduced Image", WINDOW_AUTOSIZE);
	//imshow("Reduced Image",imgOriginalIR);
	//cout <<"Ir version of reduced image is "<< imgOriginalIR << endl;

	//Mat imgReduced = constructPyramid(imgOriginal);
	//namedWindow("Reduced Image2", WINDOW_AUTOSIZE);
	//imshow("Reduced Image2",imgReduced);
	
	//cout << "values are " << (int)imgFirstFrame.at<uchar>(0, 0) << " " << (int)imgFirstFrame.at<uchar>(0, 1) << " ";
	//cout << (int)imgFirstFrame.at<uchar>(1, 0) << " " << (int)imgFirstFrame.at<uchar>(1, 1) << " " << endl;
	//int waitHere;
	//cout << "result is"<<bilinearInterpolation(imgFirstFrame, 0.01, 0.01)<<endl;
	//cin >> waitHere;

	//imgSecondFrame = onePixelDown(imgFirstFrame);
	//WINDOWX1 = WINDOWX2 = WINDOWY1 = WINDOWY2 = 10;
	//diffWholeFrame = derivativeT(imgFirstFrame, imgSecondFrame, 1, +1);
	//diffFrame = diffWholeFrame(Rect(WINDOWX1, WINDOWY1, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
	//cout << diffFrame << endl;
	//diffFrame = createDerivativeTSubPix(imgFirstFrame, imgSecondFrame, WINDOWX1+1, WINDOWY1+1, WINDOWX2, WINDOWY2);
	//cout << diffFrame << endl;
	//int waitHere;
	//cin >> waitHere;

	//imgSecondFrame = onePixelDown(imgFirstFrame);
	//WINDOWX1 = 170.01;
	//WINDOWX2 = 170;
	//WINDOWY1 = 380.01;
	//WINDOWY2 = 380;
	//dispX = dispY = 0;
	//diffWholeFrame = derivativeT(imgFirstFrame, imgSecondFrame, +dispX, +dispY);
	//diffFrame = diffWholeFrame(Rect(WINDOWX1, WINDOWY1, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
	//cout << diffFrame << endl;
	//diffFrame = createDerivativeTSubPix(imgFirstFrame, imgSecondFrame, WINDOWX1, WINDOWY1, WINDOWX2, WINDOWY2);
	//cout << diffFrame << endl;
	//int waitHere;
	//cin >> waitHere;

	//size should be an odd number
	gaussFilter = createGaussianFilter(WINDOWSIZE, SIGMA);
	//cout << "gaussFilter is " << gaussFilter << endl;
	averageFilter = createAverageFilter(WINDOWSIZE);

	for (int start = 0; start < 50; start++){
		if (corners.size() < initialCorner*0.2){
			//cout << endl << "Corners are being recalculated" << endl;
			goodFeaturesToTrack(imgFirstFrame, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
			initialCorner = corners.size();
			//cout << "Corners are recalculated. New initial corner num is " << initialCorner << endl;
		}
		imgSecondFrame = onePixelRight(imgFirstFrame);
		//imgSecondFrame = onePixelDown(imgFirstFrame);
		imgSecondFrame = onePixelRight(imgSecondFrame);
		imgSecondFrame = onePixelDown(imgSecondFrame);
		imgSecondFrame = onePixelDown(imgSecondFrame);

		//subPixelValues=createDerivativeTSubPix(imgFirstFrame,imgSecondFrame,0.5,0.5);
		//cout << subPixelValues << endl;
		//int waitHere;
		//cin >> waitHere;
		copyFirst = imgSecondFrame.clone();

		gradWholeX = gradientX(imgFirstFrame);
		gradWholeY = gradientY(imgFirstFrame);
		

		//diffWholeFrame = derivativeT(imgFirstFrame, imgSecondFrame);
		for (int i = 0; i < corners.size(); i++){
			cout << " start of for corner is " << corners.at(i).x <<"  "<<corners.at(i).y<<endl;
			if (corners.at(i).x - WINDOWSIZE / 2 < 0 || corners.at(i).x + WINDOWSIZE / 2 > imgFirstFrame.cols-1){
				//cout << endl << "inside of first if" << endl;
				cout << "corners are before: " << corners << endl;
				//corners.resize(corners.size() - 1);
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				//cout << "corners are after: " << corners << endl;
				continue;
			}
			else if (corners.at(i).y - WINDOWSIZE / 2<0 || corners.at(i).y + WINDOWSIZE / 2>imgFirstFrame.rows-1){
				//cout << endl << "inside of first if" << endl;
				cout << "corners are before: " << corners << endl;
				//corners.resize(corners.size() - 1);
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				//cout << "corners are after: " << corners << endl;
				continue;
			}
			else{
				WINDOWX1 = floor(corners.at(i).x - WINDOWSIZE / 2);
				WINDOWY1 = floor(corners.at(i).y- WINDOWSIZE / 2);

				WINDOWX2 = floor(corners.at(i).x - WINDOWSIZE / 2);
				WINDOWY2 = floor(corners.at(i).y - WINDOWSIZE / 2);
			}
			gradX = gradWholeX(Rect(WINDOWX1, WINDOWY1, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
			gradY = gradWholeY(Rect(WINDOWX1, WINDOWY1, WINDOWSIZE, WINDOWSIZE)); // using a rectangle
			//cout << "gradX is " << gradX << endl;
			//cout << "gradY is " << gradY << endl;

			Ixx = applyKernel(gradX, gradX, WINDOWSIZE);
			Ixy = applyKernel(gradX, gradY, WINDOWSIZE);
			Iyx = applyKernel(gradY, gradX, WINDOWSIZE);
			Iyy = applyKernel(gradY, gradY, WINDOWSIZE);
			//cout << "Ixx is " << Ixx << endl;
			//cout << "Ixy is " << Ixy << endl;
			//cout << "Iyx is " << Iyx << endl;
			//cout << "Iyy is " << Iyy << endl;


			G11 = applyWindowFunction(Ixx, gaussFilter);
			G12 = applyWindowFunction(Ixy, gaussFilter);
			G21 = applyWindowFunction(Iyx, gaussFilter);
			G22 = applyWindowFunction(Iyy, gaussFilter);

			if (imgFirstFrame.size() != imgSecondFrame.size()){
				cerr << "Images have different sizes" << endl;
				return -1;
			}
			dispX = dispY = 0;
			//diffWholeFrame = derivativeT(imgFirstFrame, imgSecondFrame, +dispX, +dispY);
			//diffFrame = diffWholeFrame(Rect(WINDOWY1, WINDOWX1, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
			//cout << WINDOWX1 << " " << WINDOWY1 << " " << WINDOWX2 << " " << WINDOWY2 << endl;
			//cout << diffFrame << endl;
		label:
			if (WINDOWX1+WINDOWSIZE>imgFirstFrame.cols||WINDOWX1<0){
				cout << "In the window is outside the image" << endl;
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				continue;
			}
			else if (WINDOWY1+WINDOWSIZE>imgFirstFrame.rows||WINDOWY1<0){
				cout << "In the window is outside the image" << endl;
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				continue;
			}
			else{
				diffFrame = createDerivativeTSubPix(imgFirstFrame, imgSecondFrame, WINDOWX1, WINDOWY1, WINDOWX2, WINDOWY2);
			}
			cout <<"Window boundaries "<< WINDOWX1 << " " << WINDOWY1 << " " << WINDOWX2 << " " << WINDOWY2 << endl;
			cout << "corner num is " << corners.size()<<endl;
			//cout << diffFrame << endl;
			//int waitHere;
			//cin >> waitHere;

			R1 = applyKernel(diffFrame, gradX, WINDOWSIZE);
			R2 = applyKernel(diffFrame, gradY, WINDOWSIZE);

			res1 = applyWindowFunction(R1, gaussFilter);
			res2 = applyWindowFunction(R2, gaussFilter);
			det = G22*G11 - G12*G21;
			dispX = (G22 *res1 - G12 *res2) / det;
			dispY = (G11 *res2 - G21 *res1) / det;
			cout << i << " G11 " << G11 << "	G12: " << G12;
			cout << " G21 " << G21 << "	G22: " << G22;
			cout << " res1 " << res1 << " res2 " << res2 << endl;
			cout << i << " disp x : " << dispX << "	disp y : " << dispY << endl;

			if (isnan(dispX) || isnan(dispY)){
				cout << endl << "inside of first if" << endl;
				//corners.resize(corners.size() - 1);
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				continue;
			}
			else if (corners.at(i).x + dispX<0 || corners.at(i).x + dispX >= imgFirstFrame.cols){
				cout << endl << "inside of second if over x" << endl;
				//corners.resize(corners.size()-1);
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				continue;
			}
			else if (corners.at(i).y + dispY<0 || corners.at(i).y + dispY >= imgFirstFrame.rows){
				cout << endl << "inside of second if over y" << endl;
				//corners.resize(corners.size() - 1);
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				continue;
			}
			else{
				//cout << i << " disp x : " << dispX << "	disp y : " << dispY << endl;
				corners.at(i).x += dispX;
				corners.at(i).y += dispY;
				///cout << i << ": x position " << corners.at(i).x << "	y position " << corners.at(i).y << endl;
				//cout << "windowx1 is " << WINDOWX1 << " windowy1 is " << WINDOWY1 << endl;
				if (abs(dispX) > 0.05 || abs(dispY) > 0.05){
					//cout << "dispX in the iter " << dispX << " " << "dispY in the iter " << dispY << endl;
					iterNum++;
					cout << "iteration number is " << iterNum << endl;
					//cout << "diffFrame is " << diffFrame << endl;
					if (iterNum > 10){
						iterNum = 0;
						corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
						i--;
						continue;
					}
					WINDOWX1 -=dispX;
					WINDOWY1 -=dispY;
					//cout << "windowx1 is " << WINDOWX1 << " windowy1 is " << WINDOWY1 << " windowx2 is " << WINDOWX2;
					///cout<<" windowy2 is "<<WINDOWY2<<endl;
					goto label;
				}
				iterNum = 0;
				
				//continue;
			}
			cout << " end of for corner is " << corners.at(i).x << "  " << corners.at(i).y << endl;
		}
		cvtColor(copyFirst, imgColorful, CV_GRAY2RGB);
		//cout <<"Type is" << type2str(imgColorful.type())<<endl;
		//size_t j, m;
		for (j = m = 0; j < corners.size(); j++)
		{
			corners[m++] = corners[j];
			circle(imgColorful, corners.at(j), 3, Scalar(0, 255, 0), -1, 8);
		}

		namedWindow("Video", WINDOW_AUTOSIZE);
		imshow("Video", imgColorful);
		//cout << endl << "Number of corners are " << corners.size() << endl;
		waitKey(50);
		//copyFirst = imgSecondFrame.clone();
		swap(imgFirstFrame, imgSecondFrame);
	}
	waitKey(0);
	while (true){

	}
	return 0;
}
*/


//Try on an original Video,reference

/*
int main(){
	Mat gray, prevGray, image, infrared, prevInfrared,dst;
	vector<Point2f> points[2];
	vector<uchar> status;
	vector<float> err;
	size_t i, k;
	int j = 0;
	float G11, G12, G21, G22, res1, res2, det, dispX, dispY;
	int initialCorner;
	char* videoName = "./video/planeVideoShort.mp4";
	//char* videoName = "./video/Megamind.avi";
	const int MAX_COUNT = 500;
	VideoCapture cap;

	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	Size subPixWinSize(10, 10), winSize(31, 31);

	cap.open(videoName);
	if (!cap.isOpened()){
		cout << "Could not initialize capturing...\n";
		return 0;
	}

	namedWindow("LK Demo", 1);

	for (;;){
		Mat frame;
		cap >> frame;
		if (frame.empty())
			break;

		frame.copyTo(image);
		cvtColor(image, gray, CV_BGR2GRAY);

		if (j == 0){
			// automatic initialization
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			//cornerHarris(gray, dst, 2, 3, 0.04, BORDER_DEFAULT);
			cout << "inside if " << endl;
			//cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);

			cout  << "size" << points[1].size() << endl;
			for (i = k = 0; i < points[1].size(); i++){
				cout << "inside of for" << endl;
				points[1][k++] = points[1][i];
				circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
		}
		if (!points[0].empty()){
			if (prevGray.empty()){
				gray.copyTo(prevGray);
			}
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
			3, termcrit, 0, 0.001);
			cout << status.size() << " " << err.size() << endl;
			for (i = k = 0; i < points[1].size(); i++){
				if (!status[i])
					continue;
				points[1][k++] = points[1][i];
				circle(image, points[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
			points[1].resize(k);
		}
		imshow("LK Demo", image);
		char c = (char)waitKey(50);
		if (c == 27)
			break;
		std::swap(points[1], points[0]);
		swap(prevGray, gray);
		j++;
	}
	while (true){

	}
	return 0;
}
*/

//Turn function what you wrote

/*
int main(){
	//int WINDOWX;
	//int WINDOWY;
	int initialCorner, count,returnValue;

	//char* videoName = "./video/Megamind_bugy.avi";
	//char* videoName = "./video/Megamind.avi";
	//char* videoName = "./video/planeVideoShort.mp4";
	char* videoName = "./video/plane.mp4";
	//char *videoName = "./video/cruise.mp4";
	Mat diffFrame, diffWholeFrame, R1, R2, copyFirst, imgOriginal, imgColorful, gray, imgFirstFrame, imgSecondFrame;
	size_t j, m;

	VideoCapture cap;
	cap.open(videoName);
	if (!cap.isOpened()){
		cerr << "Couldn't open the video";
		return -1;
	}

	//goodFeaturesToTrack Parameters
	vector<Point2f> corners;
	vector<Point2f> nextCorners;
	int maxCorners = 11;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;

	for (count = 0;; count++){
		Mat frame;
		cap >> frame;
		if (frame.empty()){
			cerr << "Frame is empty";
			return -1;
		}
		cvtColor(frame, gray, CV_RGB2GRAY);
		imgFirstFrame = gray.clone();
		if (count == 0){
			cout << "Size of the video frames are " << frame.rows << ", " << frame.cols << endl;
			imgOriginal = imgFirstFrame.clone();
			cout << type2str(imgFirstFrame.type()) << endl;
			goodFeaturesToTrack(imgFirstFrame, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
			if (corners.size() == 0){
				initialCorner = 1;
			}
			else{
				initialCorner = corners.size();
			}
			cout << "Size of the corner is " << corners.size() << endl;
			for (j = m = 0; j < corners.size(); j++)
			{
				corners[m++] = corners[j];
				circle(imgOriginal, corners[j], 3, Scalar(255, 255, 255), -1, 8);
			}
			cout << "corners are " << corners << endl;
			namedWindow("imgFirstFrame", WINDOW_AUTOSIZE);
			imshow("imgFirstFrame", imgOriginal);
			imgSecondFrame = imgFirstFrame.clone();
			nextCorners = corners;
		}
		returnValue=calcOpticalFlowMotion(imgFirstFrame, imgSecondFrame, corners, nextCorners);
		cout << "Size of the nextCorners is " <<nextCorners.size()<<endl<< nextCorners << endl;
		if (returnValue!=0){
			cerr << "there was a problem in calculation" << endl;
			cout << returnValue<<endl;
			while (true){

			}
			return -1;
		}
		if (nextCorners.size() <= initialCorner*0.1){
			cout << endl << "Corners are being recalculated" << endl;
			goodFeaturesToTrack(imgFirstFrame, nextCorners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
			initialCorner = nextCorners.size();
			cout << "Corners are recalculated. New initial corner num is " << initialCorner << endl;
			cout << "corners are " << corners << endl;
			corners.resize(nextCorners.size());
		}
		copyFirst = imgSecondFrame.clone();

		cvtColor(copyFirst, imgColorful, CV_GRAY2RGB);
		for (j = m = 0; j < nextCorners.size(); j++)
		{
			nextCorners[m++] = nextCorners[j];
			circle(imgColorful, nextCorners[j], 3, Scalar(0, 255, 0), -1, 8);
		}
		namedWindow("Video", WINDOW_AUTOSIZE);
		imshow("Video", imgColorful);
		cout << endl << "Number of corners are " << corners.size() << endl;
		waitKey(30);
		swap(imgFirstFrame, imgSecondFrame);
		swap(corners, nextCorners);
		cout << "frame number is " << count << endl;
	}
	waitKey(1000);
	while (true){

	}
	return 0;
}
*/


//try on a video .mp4 .avi

/*
int main(){
	int WINDOWX;
	int WINDOWY;

	//char* videoName = "./video/Megamind_bugy.avi";
	//char* videoName = "./video/Megamind.avi";
	char* videoName = "./video/planeVideoShort.mp4";
	Mat imgFirstFrame,imgSecondFrame, gradX, gradY, gradWholeX, gradWholeY, Ixx, Ixy, Iyx, Iyy, gaussFilter, averageFilter;
	Mat diffFrame, diffWholeFrame, R1, R2, copyFirst, imgOriginal, imgColorful,gray;
	size_t j, m;
	float G11, G12, G21, G22, res1, res2, det, dispX, dispY;
	int initialCorner,count;

	VideoCapture cap;
	cap.open(videoName);
	if (!cap.isOpened()){
		cerr << "Couldn't open the video";
		return -1;
	}

	//goodFeaturesToTrack Parameters
	vector<Point2f> corners;
	int maxCorners = 50;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 11;
	bool useHarrisDetector = false;
	double k = 0.04;

	gaussFilter=createGaussianFilter(WINDOWSIZE,SIGMA);
	for (count=0;;count++){
		Mat frame;
		cap >> frame;
		if (frame.empty()){
			cerr << "Frame is empty";
			return -1;
		}
		cvtColor(frame, gray, CV_RGB2GRAY);
		imgFirstFrame = gray.clone();
		if (count==0){
			cout << "Size of the video frames are " << frame.rows << ", " << frame.cols << endl;
			imgOriginal = imgFirstFrame.clone();
			cout << type2str(imgFirstFrame.type())<<endl;
			goodFeaturesToTrack(imgFirstFrame, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
			if (corners.size() == 0){
				initialCorner = 1;
			}
			else{
				initialCorner = corners.size();
			}
			cout <<"Size of the corner is "<< corners.size()<<endl;
			for (j = m = 0; j < corners.size(); j++)
			{
				corners[m++] = corners[j];
				circle(imgOriginal, corners[j], 3, Scalar(255, 255, 255), -1, 8);
			}
			cout << "corners are " << corners << endl;
			namedWindow("imgFirstFrame", WINDOW_AUTOSIZE);
			imshow("imgFirstFrame", imgOriginal);
			//waitKey(0);
			imgSecondFrame = imgFirstFrame.clone();
		}
		if (corners.size() <= initialCorner*0.3 ){
			cout << endl << "Corners are being recalculated" << endl;
			goodFeaturesToTrack(imgFirstFrame, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
			initialCorner = corners.size();
			cout << "Corners are recalculated. New initial corner num is " << initialCorner << endl;
			cout << "corners are " << corners << endl;
		}

		//if (imgSecondFrame.empty()){
		//	swap(imgFirstFrame, imgSecondFrame);
		//	continue;
		//}
		//cout <<"Boolean "<< (imgFirstFrame.data == imgSecondFrame.data) << endl;

		copyFirst = imgSecondFrame.clone();
		gradWholeX = gradientX(imgFirstFrame);
		gradWholeY = gradientY(imgFirstFrame);

		diffWholeFrame = Mat::zeros(imgFirstFrame.size(), CV_32F);
		int diff = 0;
		for (int i = 0; i < imgFirstFrame.rows; i++){
			for (int j = 0; j < imgFirstFrame.cols; j++){
				diffWholeFrame.at<float>(i, j) = (int)imgFirstFrame.at<uchar>(i, j) - (int)imgSecondFrame.at<uchar>(i, j);
			}
		}
		for (int i = 0; i < corners.size(); i++){
			if (corners.at(i).x - WINDOWSIZE / 2 < 0 || corners.at(i).x + WINDOWSIZE / 2 > imgFirstFrame.rows){
				cout << endl << "inside of first if" << endl;
				cout << "corners are before: " << corners << endl;
				//corners.resize(corners.size() - 1);
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				cout << "corners are after: " << corners << endl;
				continue;
			}
			else if (corners.at(i).y - WINDOWSIZE / 2 < 0 || corners.at(i).y + WINDOWSIZE / 2 > imgFirstFrame.cols){
				cout << endl << "inside of first if" << endl;
				cout << "corners are before: " << corners << endl;
				//corners.resize(corners.size() - 1);
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				cout << "corners are after: " << corners << endl;
				continue;
			}
			else{
				cout << "initializing window" << endl;
				WINDOWX = corners.at(i).y - WINDOWSIZE / 2;
				WINDOWY = corners.at(i).x - WINDOWSIZE / 2;
				cout << "initialized window" << corners.at(i).x + WINDOWSIZE / 2 << "  " << corners.at(i).y + WINDOWSIZE / 2 << endl;
				cout << "value is " << (int)imgFirstFrame.at<uchar>(corners.at(i).x + WINDOWSIZE / 2, corners.at(i).y + WINDOWSIZE / 2);
				cout << "check here " << endl;
			}
			//cout << "above gradX " << endl;
			gradX = gradWholeX(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
			gradY = gradWholeY(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle
			//cout << "below gradX hata var " << endl;
			Ixx = applyKernel(gradX, gradX, WINDOWSIZE);
			Ixy = applyKernel(gradX, gradY, WINDOWSIZE);
			Iyx = applyKernel(gradY, gradX, WINDOWSIZE);
			Iyy = applyKernel(gradY, gradY, WINDOWSIZE);

			//cout << "Ixx is " << Ixx<<endl;
			G11 = applyWindowFunction(Ixx, gaussFilter);
			G12 = applyWindowFunction(Ixy, gaussFilter);
			G21 = applyWindowFunction(Iyx, gaussFilter);
			G22 = applyWindowFunction(Iyy, gaussFilter);

			if (imgFirstFrame.size() != imgSecondFrame.size()){
				cerr << "Images have different sizes" << endl;
				return -1;
			}
			diffFrame = diffWholeFrame(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
			//cout << "difFrame is " << diffFrame << endl;
			R1 = applyKernel(diffFrame, gradX, WINDOWSIZE);
			R2 = applyKernel(diffFrame, gradY, WINDOWSIZE);

			res1 = applyWindowFunction(R1, gaussFilter);
			res2 = applyWindowFunction(R2, gaussFilter);
			det = G22*G11 - G12*G21;
			cout << "det is " << det << endl;
			cout << "G matrix is " << G11 << " " << G12 << " " << G21 << " " << G22 << endl;
			dispX = (G22 *res1 - G12 *res2) / det;
			dispY = (G11 *res2 - G21 *res1) / det;
			cout << i << " disp x : " << dispX << "	disp y : " << dispY << endl;

			if (isnan(dispX) || isnan(dispY)){
				cout << endl << "inside of first if isnan() " << endl;
				cout << "current value of corner is " << corners.at(i) << endl;
				//corners.resize(corners.size() - 1);
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				continue;
			}
			else if (corners.at(i).x + dispX < 0 || corners.at(i).x + dispX >= imgFirstFrame.rows){
				cout << endl << "inside of second if over x" << endl;
				//corners.resize(corners.size()-1);
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				continue;
			}
			else if (corners.at(i).y + dispY < 0 || corners.at(i).y + dispY >= imgFirstFrame.cols){
				cout << endl << "inside of second if over y" << endl;
				//corners.resize(corners.size() - 1);
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				continue;
			}
			else{
				cout << i << " disp x : " << dispX << "	disp y : " << dispY << endl;
				corners.at(i).x += dispX;
				corners.at(i).y += dispY;
				cout << i << ": x position " << corners.at(i).x << "	y position " << corners.at(i).y << endl;
				//continue;
			}

		}
		cvtColor(copyFirst, imgColorful, CV_GRAY2RGB);
		//cout <<"Type is" << type2str(imgColorful.type())<<endl;
		//size_t j, m;
		for (j = m = 0; j < corners.size(); j++)
		{
			corners[m++] = corners[j];
			circle(imgColorful, corners[j], 3, Scalar(0, 255, 0), -1, 8);
		}

		namedWindow("Video", WINDOW_AUTOSIZE);
		imshow("Video", imgColorful);
		cout << endl << "Number of corners are " << corners.size() << endl;
		waitKey(30);
		//copyFirst = imgSecondFrame.clone();
		swap(imgFirstFrame, imgSecondFrame);
		cout << "count number is " << count << endl;
	}
	waitKey(1000);
	while (true){

	}
	return 0;
}
*/

// Bugs are fixed 

/*
int main(){
	int WINDOWX;
	int WINDOWY;

	char* imName = "./images/copyLena.jpg";
	//char* imName = "./images/0thFrame.jpg";
	//char* imName = "./images/7thFrame.jpg";
	//char* imName = "./images/14thFrame.jpg";
	Mat imgSecondFrame,gradX,gradY,gradWholeX,gradWholeY,Ixx,Ixy,Iyx,Iyy,gaussFilter,averageFilter;
	Mat diffFrame,diffWholeFrame,R1,R2,copyFirst,imgOriginal,imgColorful;
	float G11, G12, G21, G22,res1,res2,det,dispX,dispY;
	int initialCorner;

	Mat imgFirstFrame = imread(imName, CV_LOAD_IMAGE_GRAYSCALE);
	imgOriginal = imgFirstFrame.clone();
	//imgSecondFrame = onePixelRight(imgFirstFrame);

	vector<Point2f> corners;
	int maxCorners = 50;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 11;
	bool useHarrisDetector = false;
	double k = 0.04;
	//GaussianBlur(imgFirstFrame, imgFirstFrame, Size(3, 3), 1, 1);
	goodFeaturesToTrack(imgFirstFrame, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
	initialCorner = corners.size();
	size_t j, m;
	for (j = m = 0; j < corners.size(); j++)
	{

		corners[m++] = corners[j];
		circle(imgOriginal, corners[j], 3, Scalar(255, 255, 255), -1, 8);
	}
	namedWindow("imgFirstFrame",WINDOW_AUTOSIZE);
	imshow("imgFirstFrame", imgOriginal);

	//size should be an odd number
	gaussFilter = createGaussianFilter(WINDOWSIZE, SIGMA);
	averageFilter = createAverageFilter(WINDOWSIZE);

	for (int start = 0; start < 50; start++){
		if (corners.size() < initialCorner*0.4){
			cout << endl << "Corners are being recalculated" << endl;
			goodFeaturesToTrack(imgFirstFrame, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
			initialCorner = corners.size();
			cout << "Corners ar recalculated. New initial corner num is "<<initialCorner << endl;
		}
		imgSecondFrame = onePixelRight(imgFirstFrame);
		imgSecondFrame = onePixelRight(imgSecondFrame);
		imgSecondFrame = onePixelDown(imgSecondFrame);
		imgSecondFrame = onePixelDown(imgSecondFrame);

		copyFirst = imgSecondFrame.clone();

		gradWholeX = gradientX(imgFirstFrame);
		gradWholeY = gradientY(imgFirstFrame);

		diffWholeFrame = Mat::zeros(imgFirstFrame.size(), CV_32F);
		int diff = 0;
		for (int i = 0; i < imgFirstFrame.rows; i++){
			for (int j = 0; j < imgFirstFrame.cols; j++){
				diffWholeFrame.at<float>(i, j) = (int)imgFirstFrame.at<uchar>(i, j) - (int)imgSecondFrame.at<uchar>(i, j);
			}
		}
		for (int i = 0; i < corners.size(); i++){
			if (corners.at(i).x - WINDOWSIZE / 2 < 0 || corners.at(i).x + WINDOWSIZE / 2 > imgFirstFrame.rows){
				cout << endl << "inside of first if" << endl;
				cout << "corners are before: " << corners << endl;
				//corners.resize(corners.size() - 1);
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				cout << "corners are after: " << corners << endl;
				continue;
			}
			else if (corners.at(i).y - WINDOWSIZE / 2<0 || corners.at(i).y + WINDOWSIZE / 2>imgFirstFrame.cols){
				cout << endl << "inside of first if" << endl;
				cout << "corners are before: " << corners << endl;
				//corners.resize(corners.size() - 1);
				corners.erase(std::remove(corners.begin(),corners.end(),corners.at(i)),corners.end());
				i--;
				cout << "corners are after: " << corners << endl;
				continue;
			}
			else{
				WINDOWX = corners.at(i).y - WINDOWSIZE / 2;
				WINDOWY = corners.at(i).x - WINDOWSIZE / 2;
			}

			gradX = gradWholeX(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
			gradY = gradWholeY(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle

			Ixx = applyKernel(gradX, gradX, WINDOWSIZE);
			Ixy = applyKernel(gradX, gradY, WINDOWSIZE);
			Iyx = applyKernel(gradY, gradX, WINDOWSIZE);
			Iyy = applyKernel(gradY, gradY, WINDOWSIZE);


			G11 = applyWindowFunction(Ixx, gaussFilter);
			G12 = applyWindowFunction(Ixy, gaussFilter);
			G21 = applyWindowFunction(Iyx, gaussFilter);
			G22 = applyWindowFunction(Iyy, gaussFilter);

			if (imgFirstFrame.size() != imgSecondFrame.size()){
				cerr << "Images have different sizes" << endl;
				return -1;
			}
			diffFrame = diffWholeFrame(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 

			R1 = applyKernel(diffFrame, gradX, WINDOWSIZE);
			R2 = applyKernel(diffFrame, gradY, WINDOWSIZE);

			res1 = applyWindowFunction(R1, gaussFilter);
			res2 = applyWindowFunction(R2, gaussFilter);
			det = G22*G11 - G12*G21;
			dispX = (G22 *res1 - G12 *res2) / det;
			dispY = (G11 *res2 - G21 *res1) / det;
			cout << i << " disp x : " << dispX << "	disp y : " << dispY << endl;

			if (isnan(dispX) || isnan(dispY)){
				cout << endl << "inside of first if" << endl;
				//corners.resize(corners.size() - 1);
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				continue;
			}
			else if (corners.at(i).x + dispX<0 || corners.at(i).x + dispX>=imgFirstFrame.rows){
				cout << endl << "inside of second if over x" << endl;
				//corners.resize(corners.size()-1);
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				continue;
			}
			else if (corners.at(i).y + dispY<0 || corners.at(i).y + dispY>=imgFirstFrame.cols){
				cout << endl << "inside of second if over y" << endl;
				//corners.resize(corners.size() - 1);
				corners.erase(std::remove(corners.begin(), corners.end(), corners.at(i)), corners.end());
				i--;
				continue;
			}
			else{
				//cout << i << " disp x : " << dispX << "	disp y : " << dispY << endl;
				corners.at(i).x += dispX;
				corners.at(i).y += dispY;
				cout << i << ": x position " << corners.at(i).x << "	y position " << corners.at(i).y << endl;
				//continue;
			}

		}
		cvtColor(copyFirst,imgColorful,CV_GRAY2RGB);
		//cout <<"Type is" << type2str(imgColorful.type())<<endl;
		//size_t j, m;
		for (j = m = 0; j < corners.size(); j++)
		{
			corners[m++] = corners[j];
			circle(imgColorful, corners[j], 3, Scalar(0,255, 0), -1, 8);
		}

		namedWindow("Video", WINDOW_AUTOSIZE);
		imshow("Video", imgColorful);
		cout << endl << "Number of corners are " << corners.size() << endl;
		waitKey(50);
		//copyFirst = imgSecondFrame.clone();
		swap(imgFirstFrame,imgSecondFrame);
	}
	waitKey(0);
	while (true){

	}
	return 0;
}
*/

/*
int main(){
	int WINDOWX = 442;
	int WINDOWY = 250;

	char* imName = "./images/copyLena.jpg";
	Mat imgSecondFrame;
	Mat imgFirstFrame = imread(imName,CV_LOAD_IMAGE_GRAYSCALE);
	//namedWindow("imgFirstFrame",WINDOW_AUTOSIZE);
	//imshow("imgFirstFrame",imgFirstFrame);
	vector<Point2f> corners; 
	int maxCorners = 11;
	double qualityLevel = 0.01; 
	double minDistance = 10; 
	int blockSize = 3; 
	bool useHarrisDetector = false; 
	double k = 0.04;
	GaussianBlur(imgFirstFrame, imgFirstFrame,Size(3, 3), 1, 1);
	goodFeaturesToTrack(imgFirstFrame, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
	cout << "corners are " << endl << corners << endl;
	WINDOWX = corners.at(1).x;
	WINDOWY = corners.at(1).y;
	//cout << "First corner is " << corners.at(0).y << endl;

	//imgSecondFrame = onePixelRight(imgFirstFrame);	//not close enough 
	//imgSecondFrame = onePixelLeft(imgFirstFrame);	//very close
	//imgSecondFrame = onePixelUp(imgFirstFrame);	//very close
	//imgSecondFrame = onePixelDown(imgFirstFrame);	//not close enough 
	//imgSecondFrame = imgFirstFrame;

	//imgSecondFrame = onePixelUp(imgFirstFrame);
	//imgSecondFrame = onePixelLeft(imgSecondFrame);	

	//imgFirstFrame = imgFirstFrame(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
	//imgSecondFrame = imgSecondFrame(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 

	//imgSecondFrameWindow(imgSecondFrame, Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 

	//imgFirstFrame = onePixelRight(imgFirstFrame);
	//imgSecondFrame = onePixelRight(imgSecondFrame);

	///Mat imgTry1 = imgFirstFrame;
	///Mat imgTry2 = Mat::zeros(imgTry1.size(), CV_8UC1);
	///for (int i = 0; i < 50; i++){
	///	imgTry2 = onePixelDown(imgTry1);
	///	swap(imgTry1,imgTry2);
	///}
	///namedWindow("imgTry", WINDOW_AUTOSIZE);
	///imshow("imgTry",imgTry2);

	Mat gradX = gradientX(imgFirstFrame);
	Mat gradY = gradientY(imgFirstFrame);
	Mat absGradX,absGradY;
	int min = 0;
	int max = 0;
	for (int i = 0; i < gradX.rows;i++){
		for (int j = 0; j < gradX.cols; j++){
			if ((int)gradX.at<float>(i,j)<min){
				min = (int)gradX.at<float>(i, j);
			}
			else if ((int)gradX.at<float>(i, j)>max){
				max = (int)gradX.at<float>(i, j);
			}
		}
	}

	cout<<"Min and max of the gradient operation to an image (0,255) " << min << endl << max<<endl;
	min = 0;
	max = 0;
	convertScaleAbs(gradX, absGradX);
	convertScaleAbs(gradY, absGradY);
	string ty1, ty2;
	ty1 = type2str(gradX.type());
	ty2 = type2str(absGradX.type());
	cout << ty1.c_str()<<endl<<ty2.c_str()<<endl;
	
	for (int i = 0; i < absGradX.rows; i++){
		for (int j = 0; j < absGradX.cols; j++){
			if ((int)absGradX.at<uchar>(i, j)<min){
				min = (int)absGradX.at<uchar>(i, j);
			}
			else if ((int)absGradX.at<uchar>(i, j)>max){
				max = (int)absGradX.at<uchar>(i, j);
			}
		}
	}

	cout <<"Min and max of the result of the convertScaleAbs function "<< min<<endl<<max<<endl;
	
	//Ixx = gradientX(gradX);
	//Ixy = gradientY(gradX);
	//Iyx = gradientX(gradY);
	//Iyy = gradientY(gradY);

	
	gradX = gradX(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
	gradY = gradY(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle

	///Mat Ixx = Mat::zeros(gradX.size(), CV_32F);
	///Mat Ixy = Mat::zeros(gradX.size(), CV_32F);
	///Mat Iyx = Mat::zeros(gradX.size(), CV_32F);
	///Mat Iyy = Mat::zeros(gradX.size(), CV_32F);

	Mat Ixx = applyKernel(gradX, gradX, WINDOWSIZE);
	cout << endl << "gradX*gradX is " << Ixx << endl;
	Mat Ixy = applyKernel(gradX, gradY, WINDOWSIZE);
	Mat Iyx = applyKernel(gradY, gradX, WINDOWSIZE);
	Mat Iyy = applyKernel(gradY, gradY, WINDOWSIZE);

	min = 0;
	max = 0;
	for (int i = 0; i < Ixx.rows; i++){
		for (int j = 0; j < Ixx.cols; j++){
			if ((int)Ixx.at<float>(i, j)<min){
				min = (int)Ixx.at<float>(i, j);
			}
			else if ((int)Ixx.at<float>(i, j)>max){
				max = (int)Ixx.at<float>(i, j);
			}
		}
	}
	cout << endl<<type2str(Ixx.type()).c_str()<<endl;
	//cout << Ixx;
	cout << "Min and max of the result of the second order derivative on an image(0,255) " << min << endl << max<<endl;
	///int i = 3;
	///int sigma = 0.5;
	///Mat dstIxx = Mat::zeros(Ixx.size(), CV_32F);
	///Mat dstIxy = Mat::zeros(Ixy.size(), CV_32F);
	///Mat dstIyx = Mat::zeros(Iyx.size(), CV_32F);
	///Mat dstIyy = Mat::zeros(Iyy.size(), CV_32F);

	///GaussianBlur(Ixx, dstIxx, Size(i, i), sigma, sigma);
	///GaussianBlur(Ixy, dstIxy, Size(i, i), sigma, sigma);
	///GaussianBlur(Iyx, dstIyx, Size(i, i), sigma, sigma);
	///GaussianBlur(Iyy, dstIyy, Size(i, i), sigma, sigma);

	//size should be an odd number
	Mat gaussFilter = createGaussianFilter(WINDOWSIZE, SIGMA);
	cout << endl<<"gauss filter is "<<gaussFilter<<endl;

	Mat averageFilter = createAverageFilter(WINDOWSIZE);
	cout << endl << "average filter is " << averageFilter << endl;

	//Ixx = Ixx(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
	double G11 = applyWindowFunction(Ixx, gaussFilter);
	cout << endl << "G11 is "<<G11<<endl;

	//Ixy = Ixy(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
	double G12 = applyWindowFunction(Ixy, gaussFilter);
	cout << endl << "G12 is " << G12 << endl;

	//Iyx = Iyx(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
	double G21 = applyWindowFunction(Iyx, gaussFilter);
	cout << endl << "G21 is " << G21 << endl;

	//Iyy = Iyy(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
	double G22 = applyWindowFunction(Iyy, gaussFilter);
	cout << endl << "G22 is " << G22 << endl;

	if (imgFirstFrame.size() != imgSecondFrame.size()){
		cerr << "Images have different sizes" << endl;
		return -1;
	}
	Mat diffFrame = Mat::zeros(imgFirstFrame.size(),CV_32F);
	int diff = 0;
	for (int i = 0; i < imgFirstFrame.rows;i++){
		for (int j = 0; j < imgFirstFrame.cols;j++){
			//diff = (int)imgFirstFrame.at<uchar>(i, j) + (int)imgSecondFrame.at<uchar>(i, j);
			//cout << " " << diff<<endl;
			diffFrame.at<float>(i, j) = (int)imgFirstFrame.at<uchar>(i, j) - (int)imgSecondFrame.at<uchar>(i, j);
		}
	}
	//diffFrame = imgFirstFrame - imgSecondFrame;  //doesn't work this way there is no negative number
	//cout << endl<<"The difference of Frames is "<<diffFrame;
	cout << endl << "Type of diff frame is " << type2str(diffFrame.type()) << endl;
	diffFrame = diffFrame(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
	cout << endl << "Type of diff frame is "<<type2str(diffFrame.type()) << endl;
	//gradX = gradX(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
	//gradY = gradY(Rect(WINDOWX, WINDOWY, WINDOWSIZE, WINDOWSIZE)); // using a rectangle 
	cout << endl << "gradX is " << gradX<<endl;
	cout << endl << "gradY is " << gradY<<endl;
	cout << endl << "The difference of Frames is " << diffFrame<<endl;

	Mat R1 = applyKernel(diffFrame,gradX,WINDOWSIZE);
	cout << endl << "Resulting window of diffFrame and gradX " <<endl<< R1<<endl;

	Mat R2 = applyKernel(diffFrame, gradY, WINDOWSIZE);
	cout << endl << "Resulting window of diffFrame and gradY "<<endl<<R2<<endl;

	double res1 = applyWindowFunction(R1, gaussFilter);
	cout << endl << "Res1 is "<<res1 << endl;

	double res2 = applyWindowFunction(R2, gaussFilter);
	cout << endl << "Res2 is "<<res2<<endl;

	double det = G22*G11 - G12*G21;
	double dispX = (G22 *res1 - G12 *res2)/det;
	cout << endl << "displacement in x direction is " << dispX << endl;

	double dispY = (G11 *res2 - G21 *res1)/det;
	cout << endl << "displacement in y direction is " << dispY << endl;

	
	waitKey(0);
	while (true){

	}
	return 0;
}
*/

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <cmath>
#include<sstream>

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

//int WINDOWX =442;
//int WINDOWY = 250;
int WINDOWSIZE = 15;
float SIGMA = 5;

//Moves input image one pixel right to construct motion
//Assumes input image is CV_8UC1 and returns an image with same 
//size and type(CV_8UC1)
Mat onePixelRight(Mat img);

//Moves input image one pixel left to construct motion
//Assumes input image is CV_8UC1 and returns an image with same 
//size and type(CV_8UC1)
Mat onePixelLeft(Mat img);

//Moves input image one pixel upward to construct motion
//Assumes input image is CV_8UC1 and returns an image with same 
//size and type(CV_8UC1)
Mat onePixelUp(Mat img);

//Moves input image one pixel downward to construct motion
//Assumes input image is CV_8UC1 and returns an image with same 
//size and type(CV_8UC1)
Mat onePixelDown(Mat img);

//Please note that this function includes many different ways to calculate 
//gradient in x direction. One can swtich between different methods by 
//commenting and uncommenting related parts.
//All different implemantations assume input image is type CV_8UC1 and 
//returns images type CV_32F. Current implementation uses Scharr method
//to find gradient in x direction
Mat gradientX(Mat img, Mat img2);

//This function does what above function(gradientX) does
// but for images have type CV_32F. Current implementation uses 
// Scharr method
Mat gradientXCV32F(Mat img, Mat img2);

//Please note that this function includes many different ways to calculate 
//gradient in y direction. One can swtich between different methods by 
//commenting and uncommenting related parts.
//All different implemantations assume input image is type CV_8UC1 and 
//returns images type CV_32F. Current implementation uses Scharr method
//to find gradient in y direction
Mat gradientY(Mat img, Mat img2);

//This function does what above function(gradientY) does
// but for images have type CV_32F. Current implementation uses 
// Scharr method
Mat gradientYCV32F(Mat img, Mat img2);

//This function calculates gradient in x direction just on a particular window
//more efficient than the above implementation(gradientX,gradientXCV32F)
//which makes calculation on the whole image. Function does calculation 
//in subpixel accuracy. function also assumes input images have type
//CV32F
Mat createGradientXSubPixCV32F(Mat imgFirst, Mat imgSecond, float column1, float row1, int windowSize);

//This function calculates gradient in x direction just on a particular window
//more efficient than the above implementation(gradientY,gradientYCV32F)
//which makes calculation on the whole image. Function does calculation 
//in subpixel accuracy. function also assumes input images have type
//CV32F
Mat createGradientYSubPixCV32F(Mat imgFirst, Mat imgSecond, float column1, float row1, int windowSize);

//This function is to better comprehend what img.type() means in opencv
//Function assumes that input is an integer between 0 and 7. Returns 
//a string saying the type of the image. Note this function is taken from
//stackoverflow
string type2str(int type);

//creates a Gaussian Filter size x size and sigma value 'sigma'
//for both x and y directions. The filter returned has type
//CV_32F. The function applies normalization meaning the sum
//of the numbers in the filter is 1
Mat createGaussianFilter(int size, float sigma);

//creates an average filter having dimensions size x size
//The returned filter has type CV_32F. The function applies
// normalization meaning the sum of the number in the filter
// is 1(All numbers in the filter is 1/(size x size))
Mat createAverageFilter(int size);

// Assumes size of the img and size of the kernel are the 
//same. Assumes both of the input images have type CV_32F
//Kernel is the weighting function to be applied to img
//Returning number is a float sum of the correlation of
//the img and kernel
float applyWindowFunction(Mat img, Mat kernel);

//Assumes inputs img and gradient have type CV_32F
//returns an image whose dimensions are 
//kernelSize x kernelSize and also has type CV_32F
//Applies correlation and puts the result corresponding 
//place in the returned image. Please note that img and
//gradient should have the same dimensions with returned
//image (kernelSize x kernelSize)
Mat applyKernel(Mat img, Mat gradient, int kernelSize);

//Assumes img has type CV_8UC1 and row and column are floats.
//Row representing row number and column representing column number.
//Calculates the intensity value at that position using
//bilinear interpolation. Calculates result by looking at the
//values at the neighboring pixels( 4 pixel) and returns a float
//number which is intensity value at the subpixel
float bilinearInterpolation(Mat img, float row, float column);

//Assumes img has type CV_32F and row and column are floats.
//Row representing row number and column representing column number.
//Calculates the intensity value at that position using
//bilinear interpolation. Calculates result by looking at the
//values at the neighboring pixels( 4 pixel) and returns a float
//number which is intensity value at the subpixel
float bilinearInterpolationCV32F(Mat img, float row, float column);

//Assumes both imgFirst and imgSecond have types CV_8UC1 
//column1 and row1 are the upperleft position of the window for
//imgFirst. column2 and row2 are the upperleft position of the
//window for imgSecond. Function takes time derivative between these
//two frames simply subtracking imgSecond  from the imgFirst. Does 
//this operation sub pixel accuracy
//Returned image has type CV_32F and has dimensions windowSize x windowSize
Mat createDerivativeTSubPix(Mat imgFirst, Mat imgSecond, float column1, float row1, float column2, float row2, int windowSize);

//Assumes both imgFirst and imgSecond have types CV_32F 
//column1 and row1 are the upperleft position of the window for
//imgFirst. column2 and row2 are the upperleft position of the
//window for imgSecond. Function takes time derivative between these
//two frames simply subtracking imgSecond  from the imgFirst. Does 
//this operation sub pixel accuracy
//Returned image has type CV_32F and has dimensions windowSize x windowSize
Mat createDerivativeTSubPixCV32F(Mat imgFirst, Mat imgSecond, float column1, float row1, float column2, float row2, int windowSize);

//This is the function version of whole calculations. Assumes 
//imgFirstFrame and  imgSecondFrame have type CV_8UC1 corners
//contains the corner positions, nextCorners contains corner
//positions after calculations function can drop some corners 
//at the nextCorners
int calcOpticalFlowMotion(Mat imgFirstFrame, Mat imgSecondFrame, vector<Point2f> &corners, vector<Point2f> &nextCorners);

// Assumes img has type CV_8UC1 and returns an image
//that has size img.rows/2 x img.cols/2, from img constructs 
//a new image that is one level above the pyramid using Gaussian
//technique. Returned image has type CV32F.
//In code I used built-in function pyrDown
Mat constructPyramid(Mat img);

// Assumes img has type CV32F and returns an image
//that has size img.rows/2 x img.cols/2, from img constructs 
//a new image that is one level above the pyramid using Gaussian
//technique. Returned image has type CV_32F also.
//In code I used built-in function pyrDown
Mat constructPyramidCV32F(Mat img);


Mat onePixelRight(Mat img){
	//Mat result = Mat::zeros(img.size(), CV_8UC1);
	Mat result;
	result.create(img.size(), CV_8UC1);
	for (int i = 0; i < img.rows; i++){
		for (int j = 1; j < img.cols; j++){
			result.at<uchar>(i, j) = (int)img.at<uchar>(i, j - 1);
			//cout << result.at<float>(i, j) << endl;
		}
	}
	return result;
}

Mat onePixelLeft(Mat img){
	//Mat result = Mat::zeros(img.size(), CV_8UC1);
	Mat result;
	result.create(img.size(), CV_8UC1);
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols - 1; j++){
			result.at<uchar>(i, j) = (int)img.at<uchar>(i, j + 1);
			//cout << result.at<float>(i, j) << endl;
		}
	}
	return result;
}

Mat onePixelDown(Mat img){
	//Mat result = Mat::zeros(img.size(), CV_8UC1);
	Mat result;
	result.create(img.size(), CV_8UC1);
	for (int i = 1; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++){
			result.at<uchar>(i, j) = (int)img.at<uchar>(i - 1, j);
			//cout << result.at<float>(i, j) << endl;
		}
	}
	return result;
}

Mat onePixelUp(Mat img){
	//Mat result = Mat::zeros(img.size(), CV_8UC1);
	Mat result;
	result.create(img.size(), CV_8UC1);
	for (int i = 0; i < img.rows - 1; i++){
		for (int j = 0; j < img.cols; j++){
			result.at<uchar>(i, j) = (int)img.at<uchar>(i + 1, j);
			//cout << result.at<float>(i, j) << endl;
		}
	}
	return result;
}

Mat gradientX(Mat img,Mat img2){
	int scale = 1;
	int delta = 0;
	int ddepth = CV_32F;
	Mat gradX = Mat::zeros(img.size(), CV_32F);

	//for (int i = 0; i < img.rows;i++){
	//	for (int j = 0; j < img.cols-1; j++){
	//		gradX.at<float>(i, j) = ((int)img.at<uchar>(i, j+1) - (int)img.at<uchar>(i, j));
	//	}
	//}

	//for (int i = 0; i < img.rows; i++){
	//	for (int j = 1; j < img.cols; j++){
	//		gradX.at<float>(i, j) = ((int)img.at<uchar>(i, j) - (int)img.at<uchar>(i, j-1));
	//	}
	//}

	//for (int i = 0; i < img.rows; i++){
	//	for (int j = 1; j < img.cols-1; j++){
	//		gradX.at<float>(i, j) = ((int)img.at<uchar>(i, j+1) - (int)img.at<uchar>(i, j - 1));
	//	}
	//}

	////using prewitt operator
	//float res;
	//for (int i = 1; i < img.rows-1; i++){
	//	for (int j = 1; j < img.cols - 1; j++){
	//		res = 0;
	//		for (int m = -1; m <=1;m++){
	//			res += ((int)img.at<uchar>(i+m, j + 1) - (int)img.at<uchar>(i+m, j - 1));
	//		}
	//		gradX.at<float>(i, j) = res / 3;
	//	}
	//}

	//Sobel(img, gradX, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
	//gradX = gradX / 8;

	//Mat imgCopy=Mat ::zeros(img.size(),CV_32F);
	////imgCopy.create(img.size(), CV_32F);
	//for (int i = 0; i < img.rows-1; i++){
	//	for (int j = 0; j < img.cols - 1; j++){
	//		for (int m = 0; m < 2; m++){				
	//			imgCopy.at<float>(i, j) += (int)img.at<uchar>(i + m, j)*-1 / 4 + (int)img.at<uchar>(i + m, j + 1) * 1 / 4;
	//		}
	//	}
	//}
	//Mat img2Copy = Mat::zeros(img2.size(), CV_32F);
	////imgCopy.create(img.size(), CV_32F);
	//for (int i = 0; i < img2.rows - 1; i++){
	//	for (int j = 0; j < img2.cols - 1; j++){
	//		for (int m = 0; m < 2; m++){
	//			img2Copy.at<float>(i, j) += (int)img2.at<uchar>(i + m, j)*-1 / 4 + (int)img2.at<uchar>(i + m, j + 1) * 1 / 4;
	//		}
	//	}
	//}
	//for (int i = 0; i < img2.rows - 1; i++){
	//	for (int j = 0; j < img2.cols - 1; j++){
	//		gradX.at<float>(i, j) += imgCopy.at<float>(i , j) + img2Copy.at<float>(i , j );
	//	}
	//}

	Scharr(img, gradX, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
	gradX = gradX / 32;

	//for (int i = 0; i < img.rows; i++){
	//	for (int j = 2; j < img.cols - 2; j++){
	//		gradX.at<float>(i, j) += (-1 * (int)img.at<uchar>(i, j - 2) + 8 * (int)img.at<uchar>(i, j - 1) - 8 * (int)img.at<uchar>(i, j + 1) + 1 * (int)img.at<uchar>(i, j + 2)) / 12;
	//	}
	//}
	//GaussianBlur(gradX, gradX, Size(5, 5), 1, 1);
	return gradX;
}

Mat gradientXCV32F(Mat img, Mat img2){
	int scale = 1;
	int delta = 0;
	int ddepth = CV_32F;
	Mat gradX = Mat::zeros(img.size(), CV_32F);
	//Mat imgCopy = Mat::zeros(img.size(), CV_32F);
	////imgCopy.create(img.size(), CV_32F);
	//for (int i = 0; i < img.rows - 1; i++){
	//	for (int j = 0; j < img.cols - 1; j++){
	//		for (int m = 0; m < 2; m++){
	//			imgCopy.at<float>(i, j) += img.at<float>(i + m, j)*-1 / 4 + img.at<float>(i + m, j + 1) * 1 / 4;
	//		}
	//	}
	//}
	//Mat img2Copy = Mat::zeros(img2.size(), CV_32F);
	////imgCopy.create(img.size(), CV_32F);
	//for (int i = 0; i < img2.rows - 1; i++){
	//	for (int j = 0; j < img2.cols - 1; j++){
	//		for (int m = 0; m < 2; m++){
	//			img2Copy.at<float>(i, j) += img2.at<float>(i + m, j)*-1 / 4 + img2.at<float>(i + m, j + 1) * 1 / 4;
	//		}
	//	}
	//}
	//for (int i = 0; i < img2.rows - 1; i++){
	//	for (int j = 0; j < img2.cols - 1; j++){
	//		gradX.at<float>(i, j) += imgCopy.at<float>(i, j) + img2Copy.at<float>(i, j);
	//	}
	//}

	Scharr(img, gradX, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
	gradX = gradX / 32;

	return gradX;
}

Mat gradientY(Mat img, Mat img2){
	int scale = 1;
	int delta = 0;
	int ddepth = CV_32F;
	Mat gradY = Mat::zeros(img.size(), CV_32F);
	//for (int i = 0; i < img.rows - 1; i++){
	//	for (int j = 0; j < img.cols; j++){
	//		gradY.at<float>(i, j) = ((int)img.at<uchar>(i + 1, j) - (int)img.at<uchar>(i, j));
	//	}
	//}

	//for (int i = 1; i < img.rows; i++){
	//	for (int j = 0; j < img.cols; j++){
	//		gradY.at<float>(i, j) = ((int)img.at<uchar>(i, j) - (int)img.at<uchar>(i-1, j));
	//	}
	//}

	//for (int i = 1; i < img.rows-1; i++){
	//	for (int j = 0; j < img.cols; j++){
	//		gradY.at<float>(i, j) = ((int)img.at<uchar>(i+1, j) - (int)img.at<uchar>(i - 1, j));
	//	}
	//}

	////using prewitt operator
	//float res;
	//for (int i = 1; i < img.rows - 1; i++){
	//	for (int j = 1; j < img.cols-1; j++){
	//		res = 0;
	//		for (int m = -1; m < 2; m++){
	//			res += ((int)img.at<uchar>(i + 1, j+m) - (int)img.at<uchar>(i - 1, j+m));
	//		}
	//		gradY.at<float>(i, j) = res / 3;
	//	}
	//}

	//Sobel(img, gradY, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
	//gradY = gradY / 8;

	//Mat imgCopy = Mat::zeros(img.size(), CV_32F);
	////imgCopy.create(img.size(), CV_32F);
	//for (int i = 0; i < img.rows - 1; i++){
	//	for (int j = 0; j < img.cols - 1; j++){
	//		for (int m = 0; m < 2; m++){
	//			imgCopy.at<float>(i, j) += (int)img.at<uchar>(i, j+m)*-1 / 4 + (int)img.at<uchar>(i +1, j + m) * 1 / 4;
	//		}
	//	}
	//}
	//Mat img2Copy = Mat::zeros(img2.size(), CV_32F);
	////imgCopy.create(img.size(), CV_32F);
	//for (int i = 0; i < img2.rows - 1; i++){
	//	for (int j = 0; j < img2.cols - 1; j++){
	//		for (int m = 0; m < 2; m++){
	//			img2Copy.at<float>(i, j) += (int)img2.at<uchar>(i , j+m)*-1 / 4 + (int)img2.at<uchar>(i + 1, j + m) * 1 / 4;
	//		}
	//	}
	//}
	//for (int i = 0; i < img2.rows - 1; i++){
	//	for (int j = 0; j < img2.cols - 1; j++){
	//		gradY.at<float>(i, j) += imgCopy.at<float>(i, j) + img2Copy.at<float>(i, j);
	//	}
	//}

	Scharr(img, gradY, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
	gradY = gradY / 32;

	//for (int i = 2; i < img.rows - 2; i++){
	//	for (int j = 0; j < img.cols; j++){
	//		gradY.at<float>(i, j) += (-(int)img.at<uchar>(i - 2, j) + 8 * (int)img.at<uchar>(i - 1, j) - 8 * (int)img.at<uchar>(i + 1, j) + (int)img.at<uchar>(i + 2, j)) / 12;
	//	}
	//}
	//GaussianBlur(gradY, gradY, Size(5, 5), 1, 1);
	return gradY;
}

Mat gradientYCV32F(Mat img, Mat img2){
	int scale = 1;
	int delta = 0;
	int ddepth = CV_32F;
	Mat gradY = Mat::zeros(img.size(), CV_32F);
	Mat imgCopy = Mat::zeros(img.size(), CV_32F);


	////imgCopy.create(img.size(), CV_32F);
	//for (int i = 0; i < img.rows - 1; i++){
	//	for (int j = 0; j < img.cols - 1; j++){
	//		for (int m = 0; m < 2; m++){
	//			imgCopy.at<float>(i, j) += img.at<float>(i, j + m)*-1 / 4 + img.at<float>(i + 1, j + m) * 1 / 4;
	//		}
	//	}
	//}
	//Mat img2Copy = Mat::zeros(img2.size(), CV_32F);
	////imgCopy.create(img.size(), CV_32F);
	//for (int i = 0; i < img2.rows - 1; i++){
	//	for (int j = 0; j < img2.cols - 1; j++){
	//		for (int m = 0; m < 2; m++){
	//			img2Copy.at<float>(i, j) += img2.at<float>(i, j + m)*-1 / 4 + img2.at<float>(i + 1, j + m) * 1 / 4;
	//		}
	//	}
	//}
	//for (int i = 0; i < img2.rows - 1; i++){
	//	for (int j = 0; j < img2.cols - 1; j++){
	//		gradY.at<float>(i, j) += imgCopy.at<float>(i, j) + img2Copy.at<float>(i, j);
	//	}
	//}

	Scharr(img, gradY, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
	gradY = gradY / 32;

	return gradY;
}


Mat createGradientXSubPixCV32F(Mat imgFirst,Mat imgSecond, float column1, float row1, int windowSize){
	int scale = 1;
	int delta = 0;
	int ddepth = CV_32F;
	Mat result;
	Mat final = Mat::zeros(windowSize, windowSize, CV_32F);
	Scharr(imgFirst(Rect(floor(column1),floor(row1),windowSize+1,windowSize+1)), result, ddepth, 1, 0, scale, delta, BORDER_DEFAULT);
	result = result / 32;
	float rowRem = row1 - floor(row1);
	float colRem = column1 - floor(column1);
	for (int i = 0; i < windowSize; i++){
		for (int j = 0; j < windowSize; j++){
			final.at<float>(i, j) = bilinearInterpolationCV32F(result, rowRem + i, colRem + j);
		}
	}
	return final;
}

Mat createGradientYSubPixCV32F(Mat imgFirst, Mat imgSecond, float column1, float row1, int windowSize){
	int scale = 1;
	int delta = 0;
	int ddepth = CV_32F;
	Mat result;
	Mat final = Mat::zeros(windowSize, windowSize, CV_32F);
	Scharr(imgFirst(Rect(floor(column1), floor(row1), windowSize + 1, windowSize + 1)), result, ddepth, 0, 1, scale, delta, BORDER_DEFAULT);
	result = result / 32;
	float rowRem = row1 - floor(row1);
	float colRem = column1 - floor(column1);
	for (int i = 0; i < windowSize; i++){
		for (int j = 0; j < windowSize; j++){
			final.at<float>(i, j) = bilinearInterpolationCV32F(result, rowRem + i, colRem + j);
		}
	}
	return final;
}

string type2str(int type){
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth){
	case CV_8U: r = "8U"; break;
	case CV_8S: r = "8s"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:	r = "User"; break;
	}
	r += "C";
	r += (chans + '0');
	return r;
}

Mat createGaussianFilter(int size, float sigma){
	//Mat result = Mat::zeros(size, size, CV_32F);
	Mat result;
	result.create(size, size, CV_32F);
	float e = 2.718;
	float sum = 0;
	float curResult = 0;
	for (int i = -size / 2; i <= size / 2; i++){
		for (int j = -size / 2; j <= size / 2; j++){
			curResult = pow(e, (-(i*i + j*j) / (2 * sigma*sigma)));
			result.at<float>(i + size / 2, j + size / 2) = curResult;
			sum += curResult;
		}
	}
	//cout << endl<<sum << endl;
	for (int i = -size / 2; i <= size / 2; i++){
		for (int j = -size / 2; j <= size / 2; j++){
			//curResult = pow(e, (-(i*i + j*j) / (2 * sigma*sigma)));
			result.at<float>(i + size / 2, j + size / 2) = result.at<float>(i + size / 2, j + size / 2) / sum;
			//sum += curResult;
		}
	}
	return result;
}

Mat createAverageFilter(int size){
	//Mat result = Mat::zeros(size, size, CV_32F);
	Mat result;
	result.create(size, size, CV_32F);

	float sum = size*size;
	for (int i = -size / 2; i <= size / 2; i++){
		for (int j = -size / 2; j <= size / 2; j++){
			result.at<float>(i + size / 2, j + size / 2) = 1 / sum;
		}
	}
	return result;
}

//applies window function at the center of an image
float applyWindowFunction(Mat img, Mat kernel){
	float result = 0;
	//int startR = img.rows / 2 - kernel.rows / 2;
	//int startR = 260;
	//int endR = img.rows / 2 + kernel.rows / 2;

	//int startC = img.cols / 2 - kernel.cols / 2;
	//int startC = 360;
	//int endC = img.cols / 2 + kernel.cols / 2;

	for (int i = 0; i < kernel.rows; i++){
		for (int j = 0; i < kernel.cols; i++){
			result += img.at<float>(i, j)*kernel.at<float>(i, j);
		}
	}
	return result;
}

Mat applyKernel(Mat img, Mat gradient, int kernelSize){
	//Mat result = Mat::zeros(kernelSize,kernelSize,CV_32F);
	Mat result;
	result.create(kernelSize, kernelSize, CV_32F);
	for (int i = 0; i < kernelSize; i++){
		for (int j = 0; j < kernelSize; j++){
			result.at<float>(i, j) = img.at<float>(i, j)*gradient.at<float>(i, j);
		}
	}
	return result;
}

float bilinearInterpolation(Mat img, float row, float column){
	//Assumes input image is uchar type
	float alfaX = row - floor(row);
	float alfaY = column - floor(column);
	float res = (1 - alfaX)*(1 - alfaY)*(int)img.at<uchar>(floor(row), floor(column));
	res += alfaX*(1 - alfaY)*(int)img.at<uchar>(floor(row) + 1, floor(column));
	res += (1 - alfaX)*alfaY*(int)img.at<uchar>(floor(row), floor(column) + 1);
	res += alfaX*alfaY*(int)img.at<uchar>(floor(row) + 1, floor(column) + 1);
	return res;
}

float bilinearInterpolationCV32F(Mat img, float row, float column){
	//Assumes input image is CV32F type
	float alfaX = row - floor(row);
	float alfaY = column - floor(column);
	float res = (1 - alfaX)*(1 - alfaY)*img.at<float>(floor(row), floor(column));
	res += alfaX*(1 - alfaY)*img.at<float>(floor(row) + 1, floor(column));
	res += (1 - alfaX)*alfaY*img.at<float>(floor(row), floor(column) + 1);
	res += alfaX*alfaY*img.at<float>(floor(row) + 1, floor(column) + 1);
	return res;
}

Mat createDerivativeTSubPix(Mat imgFirst, Mat imgSecond, float column1, float row1, float column2, float row2,int windowSize){
	Mat result = Mat::zeros(windowSize, windowSize, CV_32F);
	//Mat result;
	//result.create(WINDOWSIZE, WINDOWSIZE, CV_32F);
	//Mat convFirst = Mat::zeros(imgFirst.size(), CV_32F);
	////Mat convFirst;
	////convFirst.create(imgFirst.size(), CV_32F);
	//Mat convSecond = Mat::zeros(imgSecond.size(), CV_32F);
	////Mat convSecond;
	////convSecond.create(imgSecond.size(), CV_32F);
	//for (int i = 0; i < imgFirst.rows - 1; i++){
	//	for (int j = 0; j < imgFirst.cols - 1; j++){
	//		for (int m = 0; m < 2; m++){
	//			convFirst.at<float>(i, j) += 0.25*(int)imgFirst.at<uchar>(i + m, j) + 0.25*(int)imgFirst.at<uchar>(i + m, j + 1);
	//		}
	//	}
	//}
	//for (int i = 0; i < imgSecond.rows - 1; i++){
	//	for (int j = 0; j < imgSecond.cols - 1; j++){
	//		for (int m = 0; m < 2; m++){
	//			convSecond.at<float>(i, j) += 0.25*(int)imgSecond.at<uchar>(i + m, j) + 0.25*(int)imgSecond.at<uchar>(i + m, j + 1);
	//		}
	//	}
	//}
	//for (int i = 0; i < result.rows; i++){
	//	for (int j = 0; j < result.cols; j++){
	//		//result.at<float>(i, j) = bilinearInterpolationCV32F(convFirst, row1 + i, column1 + j) - convSecond.at<float>(floor(row2) + i, floor(column2) + j);
	//		result.at<float>(i, j) = +bilinearInterpolationCV32F(convFirst, row1 + i, column1 + j) - bilinearInterpolationCV32F(convSecond,row2+i,column2+j);
	//	}
	//}

	////Matlab uses this method
	//for (int i = 0; i < windowSize; i++){
	//	for (int j = 0; j < windowSize; j++){
	//		result.at<float>(i, j) = -bilinearInterpolation(imgFirst, row1 + i, column1 + j);
	//		result.at<float>(i, j) += (int)imgSecond.at<uchar>((int)floor(row2) + i, (int)floor(column2) + j);
	//	}
	//}
	//GaussianBlur(result, result, Size(5, 5), 1, 1);

	for (int i = 0; i < windowSize; i++){
		for (int j = 0; j < windowSize; j++){
			result.at<float>(i, j) = bilinearInterpolation(imgFirst, row1 + i, column1 + j) - bilinearInterpolation(imgSecond, row2 + i, column2 + j);
		}
	}

	return result;
}

Mat createDerivativeTSubPixCV32F(Mat imgFirst, Mat imgSecond, float column1, float row1, float column2, float row2, int windowSize){
	Mat result = Mat::zeros(windowSize, windowSize, CV_32F);
	//Mat result;
	//result.create(WINDOWSIZE, WINDOWSIZE, CV_32F);
	//Mat convFirst = Mat::zeros(imgFirst.size(), CV_32F);
	////Mat convFirst;
	////convFirst.create(imgFirst.size(), CV_32F);
	//Mat convSecond = Mat::zeros(imgSecond.size(), CV_32F);
	////Mat convSecond;
	////convSecond.create(imgSecond.size(), CV_32F);
	//for (int i = 0; i < imgFirst.rows - 1; i++){
	//	for (int j = 0; j < imgFirst.cols - 1; j++){
	//		for (int m = 0; m < 2; m++){
	//			convFirst.at<float>(i, j) += 0.25*imgFirst.at<float>(i + m, j) + 0.25*imgFirst.at<float>(i + m, j + 1);
	//		}
	//	}
	//}
	//for (int i = 0; i < imgSecond.rows - 1; i++){
	//	for (int j = 0; j < imgSecond.cols - 1; j++){
	//		for (int m = 0; m < 2; m++){
	//			convSecond.at<float>(i, j) += 0.25*imgSecond.at<float>(i + m, j) + 0.25*imgSecond.at<float>(i + m, j + 1);
	//		}
	//	}
	//}
	//for (int i = 0; i < result.rows; i++){
	//	for (int j = 0; j < result.cols; j++){
	//		//result.at<float>(i, j) = bilinearInterpolationCV32F(convFirst, row1 + i, column1 + j) - convSecond.at<float>(floor(row2) + i, floor(column2) + j);
	//		result.at<float>(i, j) = +bilinearInterpolationCV32F(convFirst, row1 + i, column1 + j) - bilinearInterpolationCV32F(convSecond, row2 + i, column2 + j);
	//	}
	//}

	////Matlab uses this method
	//for (int i = 0; i < windowSize; i++){
	//	for (int j = 0; j < windowSize; j++){
	//		result.at<float>(i, j) = -bilinearInterpolation(imgFirst, row1 + i, column1 + j);
	//		result.at<float>(i, j) += (int)imgSecond.at<uchar>((int)floor(row2) + i, (int)floor(column2) + j);
	//	}
	//}
	//GaussianBlur(result, result, Size(5, 5), 1, 1);

	for (int i = 0; i < windowSize; i++){
		for (int j = 0; j < windowSize; j++){
			result.at<float>(i, j) = bilinearInterpolationCV32F(imgFirst, row1 + i, column1 + j) - bilinearInterpolationCV32F(imgSecond, row2 + i, column2 + j);
		}
	}

	return result;
}


int calcOpticalFlowMotion(Mat imgFirstFrame, Mat imgSecondFrame, vector<Point2f> corners[], vector<Point2f> nextCorners[]){
	Mat gradX, gradY, gradWholeX, gradWholeY, Ixx, Ixy, Iyx, Iyy, averageFilter;
	Mat diffFrame, diffWholeFrame, R1, R2, copyFirst, imgOriginal, imgColorful, gray;
	size_t j, m;
	int iterNum = 0;
	float G11, G12, G21, G22, res1, res2, det, dispX, dispY,totalDispX,totalDispY;
	dispX = 0;
	dispY = 0;
	float WINDOWX1[3];
	float WINDOWY1[3];
	float WINDOWX2[3];
	float WINDOWY2[3];
	//have size three because there is 3 pyramid level,
	//these store the size of the window to look for in the algorithm
	// 0 means base level, 1 means middle level,2 means highes level
	int windows[3];
	windows[0] = 15;
	windows[1] = 7;
	windows[2] = 7;

	Mat gaussFilter[3];
	//calls createGaussianFilter function with parameter size of the 
	//gaussian kernel and sigma value to calculate kernel, these will
	//be used as weighting function in the algorithm
	gaussFilter[0] = createGaussianFilter(windows[0], SIGMA);
	gaussFilter[1] = createGaussianFilter(windows[1], SIGMA);
	gaussFilter[2] = createGaussianFilter(windows[2], SIGMA);

	//store copy of the imgSecondFrame
	copyFirst = imgSecondFrame.clone();

	//initialize 3 matrice for current frame and 3 matrice for next frame(3 since there is 3 pyramid level)
	Mat level[3], level2nd[3];
	//// if you want to work with 8 bit uchar images uncomment the lines below
	//// and comment the following lines till pyrDown
	//level[0] = imgFirstFrame.clone();
	//level2nd[0] = imgSecondFrame.clone();

	level[0].create(imgFirstFrame.size(), CV_32F);
	level2nd[0].create(imgSecondFrame.size(), CV_32F);
	//multiply first image(current frame) by 8 and store it as CV_32F
	//to simulate Infrared
	for (int i = 0; i < level[0].rows; i++){
		for (int j = 0; j < level[0].cols; j++){
			level[0].at<float>(i, j) = 8 * (int)imgFirstFrame.at<uchar>(i, j);
		}
	}
	//multiply second image(next frame) by 8 and store it as CV_32F
	//to simulate Infrared
	for (int i = 0; i < level2nd[0].rows; i++){
		for (int j = 0; j < level2nd[0].cols; j++){
			level2nd[0].at<float>(i, j) = 8 * (int)imgSecondFrame.at<uchar>(i, j);
		}
	}
	//calculate pyramid at one level up, level 1 for current frame
	pyrDown(level[0], level[1], Size(level[0].cols / 2, level[0].rows / 2));
	//calculate pyramid at one level up level 1 for next frame
	pyrDown(level2nd[0], level2nd[1], Size(level2nd[0].cols / 2, level2nd[0].rows / 2));

	//calculate pyramid at one level up, level 2 for current frame
	pyrDown(level[1], level[2], Size(level[1].cols / 2, level[1].rows / 2));
	//calculate pyramid at one level up, level 2 for next frame
	pyrDown(level2nd[1], level2nd[2], Size(level2nd[1].cols / 2, level2nd[1].rows / 2));

	//give an error message if size of the successive frames ar different
	if (imgFirstFrame.size() != imgSecondFrame.size()){
		cerr << "Images have different sizes" << endl;
		return -1;
	}

	//calculate movement for every corner in the current frame
	for (int i = 0; i < corners[0].size(); i++){
	startOfLoop:
		totalDispX = 0;	//initialize total diplacement as zero at the beginning of calculation of movement
		totalDispY = 0;	//initialize total diplacement as zero at the beginning of calculation of movement

		//calculate displacement at every pyramid level starting from highest level 2,1,0
		for (int curLevel = 2; curLevel >= 0; curLevel--){

			////calculate gradient in x direction input should be CV_32F image
			//gradWholeX = gradientXCV32F(level[curLevel], level2nd[curLevel]);
			////calculate gradient in y direction input should be CV_32F image
			//gradWholeY = gradientYCV32F(level[curLevel], level2nd[curLevel]);

			////calculate gradient in x direction input should be 8 bit uchar image
			////if input image is 8 bit uchar uncomment lines below and comment lines corresponding above
			//gradWholeX = gradientX(level[curLevel], level2nd[curLevel]);
			////calculate gradient in y direction input should be 8 bit uchar image
			////if input image is 8 bit uchar uncomment lines below and comment lines corresponding above
			//gradWholeY = gradientY(level[curLevel], level2nd[curLevel]);

			//if i is equal the corner num break from the loop
			if (i >= corners[curLevel].size()){
				break;
			}

			//if window boundaries are outside the current frame drop corrresponding corner(at all levels of the pyramid) from being corner 
			if (corners[curLevel].at(i).x - windows[curLevel] / 2 < 0 || corners[curLevel].at(i).x + windows[curLevel] / 2 > level[curLevel].cols - 1){
				//apply erasing operation at all levels of the pyramid 
				for (int start = 0; start < 3; start++){
					corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
				}
				//Since current corner is dropped from being corner start calculation for next corner 
				//to do so goto start of the loop
				goto startOfLoop;
			}
			//if window boundaries are outside the current frame drop corrresponding corner(at all levels of the pyramid) from being corner 
			else if (corners[curLevel].at(i).y - windows[curLevel] / 2<0 || corners[curLevel].at(i).y + windows[curLevel] / 2>level[curLevel].rows - 1){
				//apply erasing operation at all levels of the pyramid 
				for (int start = 0; start < 3; start++){
					corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
				}
				//Since current corner is dropped from being corner start calculation for next corner 
				//to do so goto start of the loop
				goto startOfLoop;
				continue;
			}
			else {
				//specify upper-left coordinates for the first and second image
				//x means column number, y means row number (0,0) is upper-left corner of the image
				//1 means window for first image, 2 means window for second image
				WINDOWX1[curLevel] = corners[curLevel].at(i).x - windows[curLevel] / 2 - 2 * totalDispX;
				WINDOWY1[curLevel] = corners[curLevel].at(i).y - windows[curLevel] / 2 - 2 * totalDispY;
				WINDOWX2[curLevel] = corners[curLevel].at(i).x - windows[curLevel] / 2;
				WINDOWY2[curLevel] = corners[curLevel].at(i).y - windows[curLevel] / 2;
				//reinitialize totalDispX and totalDispY as 0 new calculation will start at the one level below of pyramid
				totalDispX = totalDispY = 0;
			}
			////assign gradient values in the corresponding window to a new matrice
			////Rect parameters: 1st parameter is the column number of the upper-left coordinate
			////Rect parameters: 2nd parameter is the row number of the upper-left coordinate
			////Rect parameters: 3th parameter is the width of the columns(x) in the window
			////Rect parameters: 4th parameter is the width of the rows(y) in the window
			//gradX = gradWholeX(Rect(round(WINDOWX1[curLevel]), round(WINDOWY1[curLevel]), windows[curLevel], windows[curLevel])); // using a rectangle 
			//gradY = gradWholeY(Rect(round(WINDOWX1[curLevel]), round(WINDOWY1[curLevel]), windows[curLevel], windows[curLevel])); // using a rectangle

			//one can calculate the gradient in subpixel accuracy
			//functions below calculates gradients in sub pixel accuracy
			//and calculates gradient in just on the current window if these
			//functions are used there is no need to calculate the gradients
			//for whole image at the above
			gradX = createGradientXSubPixCV32F(level[curLevel], level2nd[curLevel], round(WINDOWX1[curLevel]), round(WINDOWY1[curLevel]), windows[curLevel]);
			gradY = createGradientYSubPixCV32F(level[curLevel], level2nd[curLevel], round(WINDOWX1[curLevel]), round(WINDOWY1[curLevel]), windows[curLevel]);

			// correlate gradients
			// applyKernel takes two input matrice which have sizes
			// windows[curLevel] x windows[curLevel] and takes the 
			// size of the window as input. since window is square we
			// give just one parameter
			Ixx = applyKernel(gradX, gradX, windows[curLevel]);
			Ixy = applyKernel(gradX, gradY, windows[curLevel]);
			Iyx = applyKernel(gradY, gradX, windows[curLevel]);
			Iyy = applyKernel(gradY, gradY, windows[curLevel]);

			//Ixx, Ixy,Iyx,Iyy are matrices we want to convert thme to numbers
			// for doing so, we apply a weighting function(gaussian kernel)
			// one can also apply average filter but gaussian filter is more appropriate
			// since central pixels in the window matter more 
			G11 = applyWindowFunction(Ixx, gaussFilter[curLevel]);
			G12 = applyWindowFunction(Ixy, gaussFilter[curLevel]);
			G21 = applyWindowFunction(Iyx, gaussFilter[curLevel]);
			G22 = applyWindowFunction(Iyy, gaussFilter[curLevel]);

			//dispX and dispY stores the displacement values at each iteration of the Lucas-Kanade
			//totalDispX and totalDispY are sum of these displacement for a pyramid level
			dispX = dispY = 0;

		label:	//at every iteration within a pyramid level code stars from here
			//upper calculations are always same for within a pyramid level
			// the change at every iteration is in the derivative with respect
			//time, and it is calculated at below

			//if window is outside the image erases corner and corresponding corners at the
			//other pyramid levels
			if (WINDOWX1[curLevel] + windows[curLevel]>level[curLevel].cols || WINDOWX1[curLevel]<0){
				for (int start = 0; start < 3; start++){
					corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
				}
				//After deleted corner starts the calculation of a new corner at the highest pyramid level
				goto startOfLoop;
			}
			//if window is outside the image erases corner and corresponding corners at the
			//other pyramid levels
			else if (WINDOWY1[curLevel] + windows[curLevel]>level[curLevel].rows || WINDOWY1[curLevel]<0){
				for (int start = 0; start < 3; start++){
					corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
				}
				//After deleted corner starts the calculation of a new corner at the highest pyramid level
				goto startOfLoop;
			}
			// if window is on the image, makes calculations to find displacement
			else{
				//function below calculates derivative in time for images 32 bit on the corresponding window
				diffFrame = createDerivativeTSubPixCV32F(level[curLevel], level2nd[curLevel], WINDOWX1[curLevel], WINDOWY1[curLevel], WINDOWX2[curLevel], WINDOWY2[curLevel], windows[curLevel]);
				/////if image is 8 bit function below should be used
				//diffFrame = createDerivativeTSubPix(level[curLevel], level2nd[curLevel], WINDOWX1[curLevel], WINDOWY1[curLevel], WINDOWX2[curLevel], WINDOWY2[curLevel], windows[curLevel]);
			}
			//correlates derivative in time and gradient in x and y direction
			//outputs are matrice
			R1 = applyKernel(diffFrame, gradX, windows[curLevel]);
			R2 = applyKernel(diffFrame, gradY, windows[curLevel]);

			//We need to represent these matrices as numbers
			//so we apply a weighting function(gaussFilter)
			// one can also use average filter
			res1 = applyWindowFunction(R1, gaussFilter[curLevel]);
			res2 = applyWindowFunction(R2, gaussFilter[curLevel]);

			//Finds determinant of the A matrice(Ax=b)
			//to solve for x 
			//A is 2 by 2 matrix [G11,G12; G21,G22]
			//x is 2 by 1 matrix [dispX;dispY]
			//b is 2 by 1 matrix [res1;res2]
			det = G22*G11 - G12*G21;

			//Finds unknowns dispX and dispY
			dispX = (G22 *res1 - G12 *res2) / det;
			dispY = (G11 *res2 - G21 *res1) / det;

			//if result is nan(not a number) drop this corner for being corner
			if (isnan(dispX) || isnan(dispY)){
				for (int start = 0; start < 3; start++){
					corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
				}
				//start a calculation for a new corner
				goto startOfLoop;
			}
			//if when displacement is applied to the corner, corner goes out of the image
			//drop this corner from being corner
			else if (corners[curLevel].at(i).x + dispX<0 || corners[curLevel].at(i).x + dispX >= level[curLevel].cols){
				for (int start = 0; start < 3; start++){
					corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
				}
				//start a calculation for a new corner
				goto startOfLoop;
			}
			//if when displacement is applied to the corner, corner goes out of the image
			//drop this corner from being corner
			else if (corners[curLevel].at(i).y + dispY<0 || corners[curLevel].at(i).y + dispY >= imgFirstFrame.rows){
				for (int start = 0; start < 3; start++){
					corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
				}
				//start a calculation for a new corner
				goto startOfLoop;
			}
			//if corner is still on the image after displacement has added
			//update corner position by adding dispX and and dispY correspoing corner
			else{
				corners[curLevel].at(i).x += dispX;
				corners[curLevel].at(i).y += dispY;
				//store total displacement for all iterations for a pyramid level
				totalDispX += dispX;
				totalDispY += dispY;
				//if current displacement found is larger than some threshold 
				//either x or y direction recalculate the displacement
				if (abs(dispX) > 0.05 || abs(dispY) > 0.05){
					//keep track of iteration number if doesn't convewrge with 
					iterNum++;
					//if displacement doesn't converge in 15 iteration
					//loop terminates and drop current corner from being corner
					// and also starts the calculation for a new corner
					if (iterNum > 15){
						//reinitialize iternum as 0
						iterNum = 0;
						//delete all corresponding corners at the other levels of the pyramid
						for (int start = 0; start < 3; start++){
							corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
						}
						//start the calculation for a new corner
						goto startOfLoop;
					}
					//update window position to capture motion for first image
					//Note the problem could be here, while updating window with the
					// motion I may be doing wrong calculations or faulty ones
					// one can recheck below and improve it
					WINDOWX1[curLevel] -= dispX;
					WINDOWY1[curLevel] -= dispY;

					//recalculate the displacement with moved window 
					//this should give better result
					goto label;
				}
				//reinitialize iterNum as 0
				iterNum = 0;
			}
			//update corner one level below the current pyramid
			//this operation can't be done at the base level
			if (curLevel > 0){
				for (int l = 0; l < corners[curLevel].size(); l++){
					corners[curLevel - 1].at(i).x = 2 * corners[curLevel].at(i).x;
					corners[curLevel - 1].at(i).y = 2 * corners[curLevel].at(i).y;
				}
			}
			//when found the movement and converged appropriately gives a message saying
			//positions of the window one can look the movement between frames from below
			//output
			if (curLevel == 0){
				cout << "	----success point------" << " windowx1 is " << WINDOWX1[curLevel] << " windowy1 is " << WINDOWY1[curLevel] << " windowx2 is " << WINDOWX2[curLevel] << " windowy2 is " << WINDOWY2[curLevel] << endl;
			}
		}
	}
	nextCorners = corners;
	cout << "inside the function " << corners[0].size() << " " << nextCorners[0].size() << endl;
	return 0;
}

Mat constructPyramid(Mat img){
	//Mat resultX = Mat::zeros(img.rows, img.cols / 2, CV_8UC1);
	//Mat result = Mat::zeros(img.rows / 2, img.cols / 2, CV_8UC1);
	////Mat kernel = (Mat_<float>(5, 5) << 1 / 16, 1 / 4, 6 / 16, 4 / 16, 1 / 16, 4 / 16, 16 / 16, 24 / 16, 16 / 16, 4 / 16, 6 / 16, 24 / 16, 36 / 16, 24 / 16, 6 / 16, 4 / 16, 16 / 16, 24 / 16, 16 / 16, 4 / 16, 1 / 16, 1 / 4, 6 / 16, 4 / 16, 1 / 16);
	//Mat kernel = createGaussianFilter(5, 1);
	//Mat kernel1DX = (Mat_<float>(1, 5) << 0.05, 0.25, 0.4, 0.25, 0.05);
	//Mat kernel1DY = (Mat_<float>(5, 1) << 0.05, 0.25, 0.4, 0.25, 0.05);


	//for (int i = 0; i < resultX.rows; i++){
	//	for (int j = 0; j < resultX.cols - 2; j++){
	//		for (int m = 0; m < 5; m++){
	//			resultX.at<uchar>(i, j + 2) += (int)img.at<uchar>(i, 2 * j + m)*kernel1DX.at<float>(0, m);
	//		}
	//		//cout << "i, j are " << i << " " << j << endl;
	//	}
	//}
	//for (int i = 0; i < result.rows - 2; i++){
	//	for (int j = 0; j < result.cols; j++){
	//		for (int m = 0; m < 5; m++){
	//			result.at<uchar>(i + 2, j) += (int)resultX.at<uchar>(2 * i + m, j)*kernel1DY.at<float>(m, 0);
	//		}
	//		//cout << "i, j are " << i << " " << j << endl;
	//	}
	//}
	Mat result1 = Mat::zeros(Size((img.rows+1)/2,(img.cols+1)/2),CV_32F);
	for (int i = 0; i < result1.rows; i++){
		for (int j = 0; j < result1.cols; j++){
			result1.at<float>(i, j) += 1 / 4 * (int)img.at<uchar>(2 * i, 2 * j);
			if (2 * i - 1 < 0)
				result1.at<float>(i, j) += 1 / 8 * (int)img.at<uchar>(2 * i, 2 * j);
			else
				result1.at<float>(i, j) += 1 / 8 * (int)img.at<uchar>(2 * i - 1, 2 * j);
			if (2*i+1>result1.rows)
				result1.at<float>(i, j) += 1 / 8 * (int)img.at<uchar>(2 * i, 2 * j);
			else
				result1.at<float>(i, j) += 1 / 8 * (int)img.at<uchar>(2 * i + 1, 2 * j);
			if (2*j-1<0)
				result1.at<float>(i, j) += 1 / 8 * (int)img.at<uchar>(2 * i, 2 * j);
			else
				result1.at<float>(i, j) += 1 / 8 * (int)img.at<uchar>(2 * i, 2 * j - 1);
			if (2*j+1>result1.cols)
				result1.at<float>(i, j) += 1 / 8 * (int)img.at<uchar>(2 * i, 2 * j);
			else
				result1.at<float>(i, j) += 1 / 8 * (int)img.at<uchar>(2 * i, 2 * j + 1);
			if (2 * i - 1 < 0 && 2 * j - 1 < 0)
				result1.at<float>(i, j) += 1 / 16 * (int)img.at<uchar>(2 * i, 2 * j);
			else
				result1.at<float>(i, j) += 1 / 16 * (int)img.at<uchar>(2 * i - 1, 2 * j - 1);
			if (2 * i + 1>result1.rows&& 2 * j + 1>result1.cols)
				result1.at<float>(i, j) += 1 / 16 * (int)img.at<uchar>(2 * i, 2 * j);
			else
				result1.at<float>(i, j) += 1 / 16 * (int)img.at<uchar>(2 * i + 1, 2 * j + 1);
			if (2 * i + 1>result1.rows&& 2 * j - 1<0)
				result1.at<float>(i, j) += 1 / 16 * (int)img.at<uchar>(2 * i, 2 * j);
			else
				result1.at<float>(i, j) += 1 / 16 * (int)img.at<uchar>(2 * i + 1, 2 * j - 1);
			if (2 * i - 1<0&& 2 * j + 1>result1.cols)
				result1.at<float>(i, j) += 1 / 16 * (int)img.at<uchar>(2 * i - 1, 2 * j + 1);
			else
				result1.at<float>(i, j) += 1 / 16 * (int)img.at<uchar>(2 * i - 1, 2 * j + 1);
		}
	}
	return result1;
}

Mat constructPyramidCV32F(Mat img){
	Mat result1 = Mat::zeros(Size((img.rows + 1) / 2, (img.cols + 1) / 2), CV_32F);
	for (int i = 0; i < result1.rows; i++){
		for (int j = 0; j < result1.cols; j++){
			result1.at<float>(i, j) += 1 / 4 * img.at<float>(2 * i, 2 * j);
			if (2 * i - 1 < 0)
				result1.at<float>(i, j) += 1 / 8 * img.at<float>(2 * i, 2 * j);
			else
				result1.at<float>(i, j) += 1 / 8 * img.at<float>(2 * i - 1, 2 * j);
			if (2 * i + 1>result1.rows)
				result1.at<float>(i, j) += 1 / 8 * img.at<float>(2 * i, 2 * j);
			else
				result1.at<float>(i, j) += 1 / 8 * img.at<float>(2 * i + 1, 2 * j);
			if (2 * j - 1<0)
				result1.at<float>(i, j) += 1 / 8 * img.at<float>(2 * i, 2 * j);
			else
				result1.at<float>(i, j) += 1 / 8 * img.at<float>(2 * i, 2 * j - 1);
			if (2 * j + 1>result1.cols)
				result1.at<float>(i, j) += 1 / 8 * img.at<float>(2 * i, 2 * j);
			else
				result1.at<float>(i, j) += 1 / 8 * img.at<float>(2 * i, 2 * j + 1);
			if (2 * i - 1 < 0 && 2 * j - 1 < 0)
				result1.at<float>(i, j) += 1 / 16 * img.at<float>(2 * i, 2 * j);
			else
				result1.at<float>(i, j) += 1 / 16 * img.at<float>(2 * i - 1, 2 * j - 1);
			if (2 * i + 1>result1.rows && 2 * j + 1>result1.cols)
				result1.at<float>(i, j) += 1 / 16 * img.at<float>(2 * i, 2 * j);
			else
				result1.at<float>(i, j) += 1 / 16 * img.at<float>(2 * i + 1, 2 * j + 1);
			if (2 * i + 1>result1.rows && 2 * j - 1<0)
				result1.at<float>(i, j) += 1 / 16 * img.at<float>(2 * i, 2 * j);
			else
				result1.at<float>(i, j) += 1 / 16 * img.at<float>(2 * i + 1, 2 * j - 1);
			if (2 * i - 1<0 && 2 * j + 1>result1.cols)
				result1.at<float>(i, j) += 1 / 16 * img.at<float>(2 * i - 1, 2 * j + 1);
			else
				result1.at<float>(i, j) += 1 / 16 * img.at<float>(2 * i - 1, 2 * j + 1);
		}
	}
	return result1;
}


//pyramid version image sliding
/*
int main(){
	float WINDOWX1[3];
	float WINDOWY1[3];

	float WINDOWX2[3];
	float WINDOWY2[3];

	char* imName = "./images/copyLena.jpg";
	//char* imName = "./images/lenaDar.jpg";
	//char* imName = "./images/lenaDar.jpg";
	//char* imName = "./images/0thFrame.jpg";
	//char* imName = "./images/7thFrame.jpg";
	//char* imName = "./images/14thFrame.jpg";
	Mat imgSecondFrame, gradX, gradY, gradWholeX, gradWholeY, Ixx, Ixy, Iyx, Iyy;
	Mat gaussFilter[3];
	Mat averageFilter[3];
	Mat diffFrame, diffWholeFrame, R1, R2, copyFirst, imgOriginal, imgColorful, subPixelValues;
	float G11, G12, G21, G22, res1, res2, det, dispX, dispY;
	int initialCorner;

	Mat imgFirstFrame = imread(imName, CV_LOAD_IMAGE_GRAYSCALE);
	imgOriginal = imgFirstFrame.clone();
	imgSecondFrame = onePixelRight(imgFirstFrame);
	imgSecondFrame = onePixelRight(imgSecondFrame);
	imgSecondFrame = onePixelDown(imgSecondFrame);
	imgSecondFrame = onePixelDown(imgSecondFrame);
	imwrite("./images/imgFirstFrame.jpg", imgFirstFrame);
	imwrite("./images/imgSecondFrame.jpg", imgSecondFrame);


	vector<Point2f> corners[3];
	int windows[3];
	int maxCorners = 20;
	float totalDispX = 0;
	float totalDispY = 0;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3;
	int iterNum = 0;
	bool useHarrisDetector = false;
	double k = 0.04;
	goodFeaturesToTrack(imgFirstFrame, corners[0], maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
	initialCorner = corners[0].size();
	size_t j, m;
	for (j = m = 0; j < corners[0].size(); j++)
	{
		corners[0][m++] = corners[0][j];
		circle(imgOriginal, corners[0][j], 3, Scalar(255, 255, 255), -1, 8);
	}
	corners[1].resize(corners[0].size());
	corners[2].resize(corners[0].size());
	for (int i = 0; i < corners[0].size();i++){
		corners[1].at(i).x = corners[0].at(i).x / 2;
		corners[1].at(i).y = corners[0].at(i).y / 2;
		corners[2].at(i).x = corners[0].at(i).x / 4;
		corners[2].at(i).y = corners[0].at(i).y / 4;
	}
	cout << "corners level 0 " << corners[0] << endl;
	cout << "corners level 1 " << corners[1] << endl;
	cout << "corners level 2 " << corners[2] << endl;
	namedWindow("imgFirstFrame", WINDOW_AUTOSIZE);
	imshow("imgFirstFrame", imgOriginal);

	windows[0] = 15;
	windows[1] = 7;
	windows[2] = 7;
	//int waitHere;
	//cin >> waitHere;

	////pyrDown works with images CV_32F
	//pyrDown(imgOriginalIR,imgOriginalIR,Size(imgOriginalIR.cols/2,imgOriginalIR.rows/2));


	//size should be an odd number
	gaussFilter[0] = createGaussianFilter(windows[0], SIGMA);
	gaussFilter[1] = createGaussianFilter(windows[1], SIGMA);
	gaussFilter[2] = createGaussianFilter(windows[2], SIGMA);

	averageFilter[0] = createAverageFilter(windows[0]);
	averageFilter[1] = createAverageFilter(windows[1]);
	averageFilter[2] = createAverageFilter(windows[2]);

	for (int count = 0; count < 50; count++){
		if (corners[0].size() < initialCorner*0.2){
			//cout << endl << "Corners are being recalculated" << endl;
			goodFeaturesToTrack(imgFirstFrame, corners[0], maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
			initialCorner = corners[0].size();
			corners[1].resize(corners[0].size());
			corners[2].resize(corners[0].size());
			for (int i = 0; i < corners[0].size(); i++){
				corners[1].at(i).x = corners[0].at(i).x / 2;
				corners[1].at(i).y = corners[0].at(i).y / 2;
				corners[2].at(i).x = corners[0].at(i).x / 4;
				corners[2].at(i).y = corners[0].at(i).y / 4;
			}
			//cout << "Corners are recalculated. New initial corner num is " << initialCorner << endl;
		}
		imgSecondFrame = onePixelRight(imgFirstFrame);
		//imgSecondFrame = onePixelDown(imgFirstFrame);
		imgSecondFrame = onePixelRight(imgSecondFrame);
		imgSecondFrame = onePixelDown(imgSecondFrame);
		imgSecondFrame = onePixelDown(imgSecondFrame);

		copyFirst = imgSecondFrame.clone();
		Mat level[3],level2nd[3];
		level[0] = imgFirstFrame.clone();
		level2nd[0] = imgSecondFrame.clone();

		pyrDown(level[0], level[1], Size(level[0].cols / 2, level[0].rows / 2));
		pyrDown(level2nd[0], level2nd[1], Size(level2nd[0].cols / 2, level2nd[0].rows / 2));

		pyrDown(level[1], level[2], Size(level[1].cols / 2, level[1].rows / 2));
		pyrDown(level2nd[1], level2nd[2], Size(level2nd[1].cols / 2, level2nd[1].rows / 2));

		if (imgFirstFrame.size() != imgSecondFrame.size()){
			cerr << "Images have different sizes" << endl;
			return -1;
		}

		for (int i = 0; i < corners[0].size(); i++){
		startOfLoop:
			totalDispX = 0;
			totalDispY = 0;
			for (int curLevel = 2; curLevel >= 0; curLevel--){
				//if (curLevel != 2){
				//	cout << " curLevel is " << curLevel << " windowx1 is " << WINDOWX1[curLevel] << " windowy1 is ";
				//	cout << WINDOWY1[curLevel] << " windowx2 is " << WINDOWX2[curLevel] << " windowy2 is " << WINDOWY2[curLevel] << endl;
				//}
				gradWholeX = gradientX(level[curLevel],level2nd[curLevel]);
				gradWholeY = gradientY(level[curLevel],level2nd[curLevel]);
				cout << "i is " << i << " curLevel is "<<curLevel<<endl;
				cout<<"size of the corner is "<<corners[curLevel].size() << endl;
				if (i>=corners[curLevel].size()){
					cout << "------before the break------- " << endl;
					break;
				}
				cout << " start of for corner is " << corners[curLevel].at(i).x << "  " << corners[curLevel].at(i).y << endl;
				cout << " totalDispX is " << totalDispX << " totalDispY is " << totalDispY << endl;
				if (corners[curLevel].at(i).x - windows[curLevel] / 2 < 0 || corners[curLevel].at(i).x + windows[curLevel] / 2 > level[curLevel].cols - 1){
					//cout << endl << "inside of first if" << endl;
					cout << "corners are before: " << corners[curLevel] << endl;
					//corners.resize(corners.size() - 1);
					for (int start = 0; start < 3; start++){
						corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
					}
					//i--;
					goto startOfLoop;
					//cout << "corners are after: " << corners << endl;
					continue;
				}
				else if (corners[curLevel].at(i).y - windows[curLevel] / 2<0 || corners[curLevel].at(i).y + windows[curLevel] / 2>level[curLevel].rows - 1){
					//cout << endl << "inside of first if" << endl;
					cout << "corners are before: " << corners << endl;
					//corners.resize(corners.size() - 1);
					for (int start = 0; start < 3; start++){
						corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
					}
					//i--;
					goto startOfLoop;
					//cout << "corners are after: " << corners << endl;
					continue;
				}
				else {
					//WINDOWX1[curLevel] = floor(corners[curLevel].at(i).x - windows[curLevel] / 2);
					//WINDOWY1[curLevel] = floor(corners[curLevel].at(i).y - windows[curLevel] / 2);
					//WINDOWX2[curLevel] = floor(corners[curLevel].at(i).x - windows[curLevel] / 2);
					//WINDOWY2[curLevel] = floor(corners[curLevel].at(i).y - windows[curLevel] / 2);
					WINDOWX1[curLevel] = corners[curLevel].at(i).x - windows[curLevel] / 2-2*totalDispX;
					WINDOWY1[curLevel] = corners[curLevel].at(i).y - windows[curLevel] / 2-2*totalDispY;
					WINDOWX2[curLevel] = corners[curLevel].at(i).x - windows[curLevel] / 2;
					WINDOWY2[curLevel] = corners[curLevel].at(i).y - windows[curLevel] / 2;
					totalDispX = totalDispY = 0;
				}
				gradX = gradWholeX(Rect(round(WINDOWX1[curLevel]), round(WINDOWY1[curLevel]), windows[curLevel], windows[curLevel])); // using a rectangle 
				gradY = gradWholeY(Rect(round(WINDOWX1[curLevel]), round(WINDOWY1[curLevel]), windows[curLevel], windows[curLevel])); // using a rectangle
				//cout << "gradX is " << gradX << endl;
				//cout << "gradY is " << gradY << endl;

				Ixx = applyKernel(gradX, gradX, windows[curLevel]);
				Ixy = applyKernel(gradX, gradY, windows[curLevel]);
				Iyx = applyKernel(gradY, gradX, windows[curLevel]);
				Iyy = applyKernel(gradY, gradY, windows[curLevel]);
				//cout << "Ixx is " << Ixx << endl;
				//cout << "Ixy is " << Ixy << endl;
				//cout << "Iyx is " << Iyx << endl;
				//cout << "Iyy is " << Iyy << endl;


				G11 = applyWindowFunction(Ixx, gaussFilter[curLevel]);
				G12 = applyWindowFunction(Ixy, gaussFilter[curLevel]);
				G21 = applyWindowFunction(Iyx, gaussFilter[curLevel]);
				G22 = applyWindowFunction(Iyy, gaussFilter[curLevel]);

				dispX = dispY = 0;
				
			label:
				if (WINDOWX1[curLevel] + windows[curLevel]>level[curLevel].cols || WINDOWX1[curLevel]<0){
					cout << "In the window is outside the image" << endl;
					for (int start = 0; start < 3; start++){
						corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
					}
					//i--;
					goto startOfLoop;
					continue;
				}
				else if (WINDOWY1[curLevel] + windows[curLevel]>level[curLevel].rows || WINDOWY1[curLevel]<0){
					cout << "In the window is outside the image" << endl;
					for (int start = 0; start < 3; start++){
						corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
					}
					//i--;
					goto startOfLoop;
					continue;
				}
				else{
					diffFrame = createDerivativeTSubPix(level[curLevel], level2nd[curLevel], WINDOWX1[curLevel], WINDOWY1[curLevel], WINDOWX2[curLevel], WINDOWY2[curLevel], windows[curLevel]);
				}
				cout << "Window boundaries " << WINDOWX1[curLevel] << " " << WINDOWY1[curLevel] << " " << WINDOWX2[curLevel] << " " << WINDOWY2[curLevel] << endl;
				cout << "corner num is " << corners[curLevel].size() << endl;
				//cout << diffFrame << endl;
				//int waitHere;
				//cin >> waitHere;

				R1 = applyKernel(diffFrame, gradX, windows[curLevel]);
				R2 = applyKernel(diffFrame, gradY, windows[curLevel]);

				res1 = applyWindowFunction(R1, gaussFilter[curLevel]);
				res2 = applyWindowFunction(R2, gaussFilter[curLevel]);
				det = G22*G11 - G12*G21;
				dispX = (G22 *res1 - G12 *res2) / det;
				dispY = (G11 *res2 - G21 *res1) / det;
				cout << i << " G11 " << G11 << "	G12: " << G12;
				cout << " G21 " << G21 << "	G22: " << G22;
				cout << " res1 " << res1 << " res2 " << res2 << endl;
				cout << i << " disp x : " << dispX << "	disp y : " << dispY << endl;

				if (isnan(dispX) || isnan(dispY)){
					cout << endl << "inside of first if" << endl;
					//corners.resize(corners.size() - 1);
					for (int start = 0; start < 3; start++){
						corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
					}
					//i--;
					goto startOfLoop;
					continue;
				}
				else if (corners[curLevel].at(i).x + dispX<0 || corners[curLevel].at(i).x + dispX >= level[curLevel].cols){
					cout << endl << "inside of second if over x" << endl;
					//corners.resize(corners.size()-1);
					for (int start = 0; start < 3; start++){
						corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
					}
					//i--;
					goto startOfLoop;
					continue;
				}
				else if (corners[curLevel].at(i).y + dispY<0 || corners[curLevel].at(i).y + dispY >= imgFirstFrame.rows){
					cout << endl << "inside of second if over y" << endl;
					//corners.resize(corners.size() - 1);
					for (int start = 0; start < 3; start++){
						corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
					}
					//i--;
					goto startOfLoop;
					continue;
				}
				else{
					//cout << i << " disp x : " << dispX << "	disp y : " << dispY << endl;
					corners[curLevel].at(i).x += dispX;
					corners[curLevel].at(i).y += dispY;
					totalDispX += dispX;
					totalDispY += dispY;
					///cout << i << ": x position " << corners.at(i).x << "	y position " << corners.at(i).y << endl;
					//cout << "windowx1 is " << WINDOWX1[curLevel] << " windowy1 is " << WINDOWY1[curLevel] << endl;
					if (abs(dispX) > 0.05 || abs(dispY) > 0.05){
						//cout << "dispX in the iter " << dispX << " " << "dispY in the iter " << dispY << endl;
						iterNum++;
						cout << "iteration number is " << iterNum << endl;
						//cout << "diffFrame is " << diffFrame << endl;
						if (iterNum > 15){
							iterNum = 0;
							for (int start = 0; start < 3; start++){
								corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
							}
							//i--;
							goto startOfLoop;
							continue;
						}
						WINDOWX1[curLevel] -= dispX;
						WINDOWY1[curLevel] -= dispY;
						//cout << "windowx1 is " << WINDOWX1[curLevel] << " windowy1 is " << WINDOWY1[curLevel] << " windowx2 is " << WINDOWX2[curLevel];
						///cout<<" windowy2 is "<<WINDOWY2[curLevel]<<endl;
						goto label;
					}
					iterNum = 0;

					//continue;
				}
				cout << " end of for corner is " << corners[curLevel].at(i).x << "  " << corners[curLevel].at(i).y << endl;
				if (curLevel > 0){
					for (int l = 0; l < corners[curLevel].size(); l++){
						corners[curLevel - 1].at(i).x = 2 * corners[curLevel].at(i).x;
						corners[curLevel - 1].at(i).y = 2 * corners[curLevel].at(i).y;
					}
				}
				cout << " curLevel is " << curLevel << " windowx1 is " << WINDOWX1[curLevel] << " windowy1 is ";
				cout<< WINDOWY1[curLevel] << " windowx2 is " << WINDOWX2[curLevel] << " windowy2 is " << WINDOWY2[curLevel] << endl;
			}
		}
		
		cvtColor(copyFirst, imgColorful, CV_GRAY2RGB);
		//cout <<"Type is" << type2str(imgColorful.type())<<endl;
		//size_t j, m;
		for (j = m = 0; j < corners[0].size(); j++)
		{
			corners[m++] = corners[j];
			circle(imgColorful, corners[0].at(j), 3, Scalar(0, 255, 0), -1, 8);
		}

		namedWindow("Video", WINDOW_AUTOSIZE);
		imshow("Video", imgColorful);
		cout <<endl<< endl << "-------------Number of corners are--------------- " << corners[0].size() << endl<<endl;
		cout << endl << endl << "---------------Current frame is: " << count << endl << endl;;
		waitKey(50);
		//copyFirst = imgSecondFrame.clone();
		swap(imgFirstFrame, imgSecondFrame);
	}
	waitKey(0);
	while (true){

	}
	return 0;
}
*/

//try on a video

int main(){
	//WINDOWX1,WINDOWX2 represents upper-left corner of the current window for current frame
	float WINDOWX1[3];	//have size 3, because in implementation there is 3 pyramid levels
	float WINDOWY1[3];	//have size 3, because in implementation there is 3 pyramid levels

	//WINDOWX1,WINDOWX2 represents upper-left corner of the current window for next frame
	float WINDOWX2[3];	//have size 3, because in implementation there is 3 pyramid levels
	float WINDOWY2[3];	//have size 3, because in implementation there is 3 pyramid levels

	//assign video directory to a string
	char* videoName = "./video/plane.mp4";
	VideoCapture cap;

	//open the video
	cap.open(videoName);

	//if cannot open the video give an error message
	if (!cap.isOpened()){
		cerr << "couldn't open the video";
		return -1;
	}
	//initialize the matrice that will be used
	Mat imgSecondFrame,imgFirstFrame, gradX, gradY, gradWholeX, gradWholeY, Ixx, Ixy, Iyx, Iyy;

	//initializes 3 gaussian filter bacause pyramid level is 3, gaussFilter is used as weighting function
	Mat gaussFilter[3];

	//initilizes 3 average filter because pyramid level is 3, currently I use gaussFilter
	//as weighting function, one can use averageFilter also
	Mat averageFilter[3];
	Mat diffFrame, diffWholeFrame, R1, R2, copyFirst, imgOriginal, imgColorful, subPixelValues;
	float G11, G12, G21, G22, res1, res2, det, dispX, dispY;
	int initialCorner;

	//initialize three vector for three pyramid level
	vector<Point2f> corners[3];

	//goodFeaturesToTrack parameters
	int maxCorners = 20;
	double qualityLevel = 0.01;
	double minDistance = 10;
	int blockSize = 3;
	int iterNum = 0;
	bool useHarrisDetector = false;
	double k = 0.04;

	size_t j, m;

	//to store total displacement at one level of the pyramid
	float totalDispX = 0;
	float totalDispY = 0;

	//have size three because there is 3 pyramid level,
	//these store the size of the window to look for in the algorithm
	// 0 means base level, 1 means middle level,2 means highes level
	int windows[3];
	windows[0] = 15;
	windows[1] = 7;
	windows[2] = 7;


	//calls createGaussianFilter function with parameter size of the 
	//gaussian kernel and sigma value to calculate kernel, these will
	//be used as weighting function in the algorithm
	gaussFilter[0] = createGaussianFilter(windows[0], SIGMA);
	gaussFilter[1] = createGaussianFilter(windows[1], SIGMA);
	gaussFilter[2] = createGaussianFilter(windows[2], SIGMA);

	//// If user wants to use average filter as weighting function 
	//// should uncomment below lines and turn the gaussFilter to
	//// averageFilter
	//averageFilter[0] = createAverageFilter(windows[0]);
	//averageFilter[1] = createAverageFilter(windows[1]);
	//averageFilter[2] = createAverageFilter(windows[2]);
	int countCornerCalc = 0;

	// count holds current frame number for
	// loop continues as long as video frames continue.
	for (int count = 0; ; count++){
		//Assign current frame to the imgFirstFrame
		cap >> imgFirstFrame;

		//Convert RGB image to the gray now imgFirstFrame is GRAYLEVEL
		cvtColor(imgFirstFrame, imgFirstFrame, CV_RGB2GRAY);

		//If count ==0(at the first frame) calculate corners
		if (count==0){

			//calculates corners in the image and stores it at the array corners[0]
			goodFeaturesToTrack(imgFirstFrame, corners[0], maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
			
			//store number of corners initially
			initialCorner = corners[0].size();

			//take a copy of the first frame
			imgOriginal = imgFirstFrame.clone();

			for (j =  0; j < corners[0].size(); j++)
			{
				//draw circles around the corners on the copy of the first frame(imgOriginal)
				circle(imgOriginal, corners[0][j], 3, Scalar(255, 255, 255), -1, 8);
			}
			//initialize size of the corners at the other pyramid levels, same as the 
			//size of the corner at the base level(output of the goodFeaturesToTrack) 
			corners[1].resize(corners[0].size());
			corners[2].resize(corners[0].size());

			//initialize corner values at the other pyramid levels
			//Since at each level size drops by 2 corresponing coordinates 
			//decreases to the half
			for (int i = 0; i < corners[0].size(); i++){
				corners[1].at(i).x = corners[0].at(i).x / 2;
				corners[1].at(i).y = corners[0].at(i).y / 2;
				corners[2].at(i).x = corners[0].at(i).x / 4;
				corners[2].at(i).y = corners[0].at(i).y / 4;
			}
			//show the the of the first frame with corners circled on the screen
			namedWindow("imgFirstFrame", WINDOW_AUTOSIZE);
			imshow("imgFirstFrame", imgOriginal);

			//since currently there is only one frame from the video(we ar at the first frame)
			//assign imgFirstFrame to the imgSecondFrame
			imgSecondFrame = imgFirstFrame.clone();
		}
		//if corner num decreases below the 1/5 of initial value recalculate the corners using 
		//goodFeaturesToTrack
		if (corners[0].size() < initialCorner*0.2){
			countCornerCalc++;	//keep track of how many times corner calculated
			//below operations are same as the calculation at the beginning
			goodFeaturesToTrack(imgFirstFrame, corners[0], maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
			initialCorner = corners[0].size();
			corners[1].resize(corners[0].size());
			corners[2].resize(corners[0].size());
			for (int i = 0; i < corners[0].size(); i++){
				corners[1].at(i).x = corners[0].at(i).x / 2;
				corners[1].at(i).y = corners[0].at(i).y / 2;
				corners[2].at(i).x = corners[0].at(i).x / 4;
				corners[2].at(i).y = corners[0].at(i).y / 4;
			}
		}
		//store copy of the imgSecondFrame
		copyFirst = imgSecondFrame.clone();

		//initialize 3 matrice for current frame and 3 matrice for next frame(3 since there is 3 pyramid level)
		Mat level[3], level2nd[3];
		//// if you want to work with 8 bit uchar images uncomment the lines below
		//// and comment the following lines till pyrDown
		//level[0] = imgFirstFrame.clone();
		//level2nd[0] = imgSecondFrame.clone();

		level[0].create(imgFirstFrame.size(), CV_32F);
		level2nd[0].create(imgSecondFrame.size(), CV_32F);
		//multiply first image(current frame) by 8 and store it as CV_32F
		//to simulate Infrared
		for (int i = 0; i < level[0].rows;i++){
			for (int j = 0; j < level[0].cols;j++){
				level[0].at<float>(i, j) = 8 * (int)imgFirstFrame.at<uchar>(i, j);
			}
		}
		//multiply second image(next frame) by 8 and store it as CV_32F
		//to simulate Infrared
		for (int i = 0; i < level2nd[0].rows; i++){
			for (int j = 0; j < level2nd[0].cols; j++){
				level2nd[0].at<float>(i, j) = 8 * (int)imgSecondFrame.at<uchar>(i, j);
			}
		}
		//calculate pyramid at one level up, level 1 for current frame
		pyrDown(level[0], level[1], Size(level[0].cols / 2, level[0].rows / 2));
		//calculate pyramid at one level up level 1 for next frame
		pyrDown(level2nd[0], level2nd[1], Size(level2nd[0].cols / 2, level2nd[0].rows / 2));

		//calculate pyramid at one level up, level 2 for current frame
		pyrDown(level[1], level[2], Size(level[1].cols / 2, level[1].rows / 2));
		//calculate pyramid at one level up, level 2 for next frame
		pyrDown(level2nd[1], level2nd[2], Size(level2nd[1].cols / 2, level2nd[1].rows / 2));

		//give an error message if size of the successive frames ar different
		if (imgFirstFrame.size() != imgSecondFrame.size()){
			cerr << "Images have different sizes" << endl;
			return -1;
		}

		//calculate movement for every corner in the current frame
		for (int i = 0; i < corners[0].size(); i++){
		startOfLoop:	
			totalDispX = 0;	//initialize total diplacement as zero at the beginning of calculation of movement
			totalDispY = 0;	//initialize total diplacement as zero at the beginning of calculation of movement

			//calculate displacement at every pyramid level starting from highest level 2,1,0
			for (int curLevel = 2; curLevel >= 0; curLevel--){
				
				////calculate gradient in x direction input should be CV_32F image
				//gradWholeX = gradientXCV32F(level[curLevel], level2nd[curLevel]);
				////calculate gradient in y direction input should be CV_32F image
				//gradWholeY = gradientYCV32F(level[curLevel], level2nd[curLevel]);

				////calculate gradient in x direction input should be 8 bit uchar image
				////if input image is 8 bit uchar uncomment lines below and comment lines corresponding above
				//gradWholeX = gradientX(level[curLevel], level2nd[curLevel]);
				////calculate gradient in y direction input should be 8 bit uchar image
				////if input image is 8 bit uchar uncomment lines below and comment lines corresponding above
				//gradWholeY = gradientY(level[curLevel], level2nd[curLevel]);

				//if i is equal the corner num break from the loop
				if (i >= corners[curLevel].size()){
					break;
				}

				//if window boundaries are outside the current frame drop corrresponding corner(at all levels of the pyramid) from being corner 
				if (corners[curLevel].at(i).x - windows[curLevel] / 2 < 0 || corners[curLevel].at(i).x + windows[curLevel] / 2 > level[curLevel].cols - 1){
					//apply erasing operation at all levels of the pyramid 
					for (int start = 0; start < 3; start++){
						corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
					}
					//Since current corner is dropped from being corner start calculation for next corner 
					//to do so goto start of the loop
					goto startOfLoop;
				}
				//if window boundaries are outside the current frame drop corrresponding corner(at all levels of the pyramid) from being corner 
				else if (corners[curLevel].at(i).y - windows[curLevel] / 2<0 || corners[curLevel].at(i).y + windows[curLevel] / 2>level[curLevel].rows - 1){
					//apply erasing operation at all levels of the pyramid 
					for (int start = 0; start < 3; start++){
						corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
					}
					//Since current corner is dropped from being corner start calculation for next corner 
					//to do so goto start of the loop
					goto startOfLoop;
					continue;
				}
				else {
					//specify upper-left coordinates for the first and second image
					//x means column number, y means row number (0,0) is upper-left corner of the image
					//1 means window for first image, 2 means window for second image
					WINDOWX1[curLevel] = corners[curLevel].at(i).x - windows[curLevel] / 2 - 2 * totalDispX;
					WINDOWY1[curLevel] = corners[curLevel].at(i).y - windows[curLevel] / 2 - 2 * totalDispY;
					WINDOWX2[curLevel] = corners[curLevel].at(i).x - windows[curLevel] / 2 ;
					WINDOWY2[curLevel] = corners[curLevel].at(i).y - windows[curLevel] / 2 ;
					//reinitialize totalDispX and totalDispY as 0 new calculation will start at the one level below of pyramid
					totalDispX = totalDispY = 0;
				}
				////assign gradient values in the corresponding window to a new matrice
				////Rect parameters: 1st parameter is the column number of the upper-left coordinate
				////Rect parameters: 2nd parameter is the row number of the upper-left coordinate
				////Rect parameters: 3th parameter is the width of the columns(x) in the window
				////Rect parameters: 4th parameter is the width of the rows(y) in the window
				//gradX = gradWholeX(Rect(round(WINDOWX1[curLevel]), round(WINDOWY1[curLevel]), windows[curLevel], windows[curLevel])); // using a rectangle 
				//gradY = gradWholeY(Rect(round(WINDOWX1[curLevel]), round(WINDOWY1[curLevel]), windows[curLevel], windows[curLevel])); // using a rectangle
				
				//one can calculate the gradient in subpixel accuracy
				//functions below calculates gradients in sub pixel accuracy
				//and calculates gradient in just on the current window if these
				//functions are used there is no need to calculate the gradients
				//for whole image at the above
				gradX = createGradientXSubPixCV32F(level[curLevel],level2nd[curLevel], round(WINDOWX1[curLevel]), round(WINDOWY1[curLevel]), windows[curLevel]);
				gradY = createGradientYSubPixCV32F(level[curLevel],level2nd[curLevel], round(WINDOWX1[curLevel]), round(WINDOWY1[curLevel]), windows[curLevel]);

				// correlate gradients
				// applyKernel takes two input matrice which have sizes
				// windows[curLevel] x windows[curLevel] and takes the 
				// size of the window as input. since window is square we
				// give just one parameter
				Ixx = applyKernel(gradX, gradX, windows[curLevel]);
				Ixy = applyKernel(gradX, gradY, windows[curLevel]);
				Iyx = applyKernel(gradY, gradX, windows[curLevel]);
				Iyy = applyKernel(gradY, gradY, windows[curLevel]);

				//Ixx, Ixy,Iyx,Iyy are matrices we want to convert thme to numbers
				// for doing so, we apply a weighting function(gaussian kernel)
				// one can also apply average filter but gaussian filter is more appropriate
				// since central pixels in the window matter more 
				G11 = applyWindowFunction(Ixx, gaussFilter[curLevel]);
				G12 = applyWindowFunction(Ixy, gaussFilter[curLevel]);
				G21 = applyWindowFunction(Iyx, gaussFilter[curLevel]);
				G22 = applyWindowFunction(Iyy, gaussFilter[curLevel]);

				//dispX and dispY stores the displacement values at each iteration of the Lucas-Kanade
				//totalDispX and totalDispY are sum of these displacement for a pyramid level
				dispX = dispY = 0;

			label:	//at every iteration within a pyramid level code stars from here
					//upper calculations are always same for within a pyramid level
					// the change at every iteration is in the derivative with respect
					//time, and it is calculated at below

				//if window is outside the image erases corner and corresponding corners at the
				//other pyramid levels
				if (WINDOWX1[curLevel] + windows[curLevel]>level[curLevel].cols || WINDOWX1[curLevel]<0){
					for (int start = 0; start < 3; start++){
						corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
					}
					//After deleted corner starts the calculation of a new corner at the highest pyramid level
					goto startOfLoop;
				}
				//if window is outside the image erases corner and corresponding corners at the
				//other pyramid levels
				else if (WINDOWY1[curLevel] + windows[curLevel]>level[curLevel].rows || WINDOWY1[curLevel]<0){
					for (int start = 0; start < 3; start++){
						corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
					}
					//After deleted corner starts the calculation of a new corner at the highest pyramid level
					goto startOfLoop;
				}
				// if window is on the image, makes calculations to find displacement
				else{
					//function below calculates derivative in time for images 32 bit on the corresponding window
					diffFrame = createDerivativeTSubPixCV32F(level[curLevel], level2nd[curLevel], WINDOWX1[curLevel], WINDOWY1[curLevel], WINDOWX2[curLevel], WINDOWY2[curLevel], windows[curLevel]);
					/////if image is 8 bit function below should be used
					//diffFrame = createDerivativeTSubPix(level[curLevel], level2nd[curLevel], WINDOWX1[curLevel], WINDOWY1[curLevel], WINDOWX2[curLevel], WINDOWY2[curLevel], windows[curLevel]);
				}
				//correlates derivative in time and gradient in x and y direction
				//outputs are matrice
				R1 = applyKernel(diffFrame, gradX, windows[curLevel]);
				R2 = applyKernel(diffFrame, gradY, windows[curLevel]);

				//We need to represent these matrices as numbers
				//so we apply a weighting function(gaussFilter)
				// one can also use average filter
				res1 = applyWindowFunction(R1, gaussFilter[curLevel]);
				res2 = applyWindowFunction(R2, gaussFilter[curLevel]);

				//Finds determinant of the A matrice(Ax=b)
				//to solve for x 
				//A is 2 by 2 matrix [G11,G12; G21,G22]
				//x is 2 by 1 matrix [dispX;dispY]
				//b is 2 by 1 matrix [res1;res2]
				det = G22*G11 - G12*G21;

				//Finds unknowns dispX and dispY
				dispX = (G22 *res1 - G12 *res2) / det;
				dispY = (G11 *res2 - G21 *res1) / det;

				//if result is nan(not a number) drop this corner for being corner
				if (isnan(dispX) || isnan(dispY)){
					for (int start = 0; start < 3; start++){
						corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
					}
					//start a calculation for a new corner
					goto startOfLoop;
				}
				//if when displacement is applied to the corner, corner goes out of the image
				//drop this corner from being corner
				else if (corners[curLevel].at(i).x + dispX<0 || corners[curLevel].at(i).x + dispX >= level[curLevel].cols){
					for (int start = 0; start < 3; start++){
						corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
					}
					//start a calculation for a new corner
					goto startOfLoop;
				}
				//if when displacement is applied to the corner, corner goes out of the image
				//drop this corner from being corner
				else if (corners[curLevel].at(i).y + dispY<0 || corners[curLevel].at(i).y + dispY >= imgFirstFrame.rows){
					for (int start = 0; start < 3; start++){
						corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
					}
					//start a calculation for a new corner
					goto startOfLoop;
				}
				//if corner is still on the image after displacement has added
				//update corner position by adding dispX and and dispY correspoing corner
				else{
					corners[curLevel].at(i).x += dispX;
					corners[curLevel].at(i).y += dispY;
					//store total displacement for all iterations for a pyramid level
					totalDispX += dispX;
					totalDispY += dispY;
					//if current displacement found is larger than some threshold 
					//either x or y direction recalculate the displacement
					if (abs(dispX) > 0.05 || abs(dispY) > 0.05){
						//keep track of iteration number if doesn't convewrge with 
						iterNum++;
						//if displacement doesn't converge in 15 iteration
						//loop terminates and drop current corner from being corner
						// and also starts the calculation for a new corner
						if (iterNum > 15){
							//reinitialize iternum as 0
							iterNum = 0;
							//delete all corresponding corners at the other levels of the pyramid
							for (int start = 0; start < 3; start++){
								corners[start].erase(std::remove(corners[start].begin(), corners[start].end(), corners[start].at(i)), corners[start].end());
							}
							//start the calculation for a new corner
							goto startOfLoop;
						}
						//update window position to capture motion for first image
						//Note the problem could be here, while updating window with the
						// motion I may be doing wrong calculations or faulty ones
						// one can recheck below and improve it
						WINDOWX1[curLevel] -= dispX;
						WINDOWY1[curLevel] -= dispY;

						//recalculate the displacement with moved window 
						//this should give better result
						goto label;
					}
					//reinitialize iterNum as 0
					iterNum = 0;
				}
				//update corner one level below the current pyramid
				//this operation can't be done at the base level
				if (curLevel > 0){
					for (int l = 0; l < corners[curLevel].size(); l++){
						corners[curLevel - 1].at(i).x = 2 * corners[curLevel].at(i).x;
						corners[curLevel - 1].at(i).y = 2 * corners[curLevel].at(i).y;
					}
				}
				//when found the movement and converged appropriately gives a message saying
				//positions of the window one can look the movement between frames from below
				//output
				if (curLevel == 0){
					cout << "	----success point------" << " windowx1 is " << WINDOWX1[curLevel] << " windowy1 is " << WINDOWY1[curLevel] << " windowx2 is " << WINDOWX2[curLevel] << " windowy2 is " << WINDOWY2[curLevel] << endl;
				}
			}
		}
		//convert grayscale image to the colorful(RGB) image
		cvtColor(copyFirst, imgColorful, CV_GRAY2RGB);
		//draw green circles around the corners
		//these green circles would visualize as tracked points in the video
		for (j = m = 0; j < corners[0].size(); j++)
		{
			corners[m++] = corners[j];
			circle(imgColorful, corners[0].at(j), 3, Scalar(0, 255, 0), -1, 8);
		}
		//show current frame with corners are green marked
		namedWindow("Video", WINDOW_AUTOSIZE);
		imshow("Video", imgColorful);
		//print out the number of corners in the frame
		//current frame number and also how many times corners are recalculated
		//to gain insight how well algorithm is working
		cout << endl << endl << "-------------Number of corners are--------------- " << corners[0].size() << endl << endl;
		cout << endl << endl << "---------------Current frame is: " << count <<", "<< countCornerCalc << "times corner calculated" << endl << endl;;
		//specifies how many mili-seconds to wait between frames
		waitKey(10);
		//swap the current frame and next frame so that 
		//at the next iteration next frame will be current frame
		swap(imgFirstFrame, imgSecondFrame);
	}
	waitKey(0);

	return 0;
}
