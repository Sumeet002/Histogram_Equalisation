#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <sys/time.h>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{

	static Mat I_RGB;
	static Mat O_RGB;
	
	I_RGB=imread("sample2.jpg");
	int scale = 2 ;
	Size size(I_RGB.cols/scale,I_RGB.rows/scale);
	int imgSize= (I_RGB.cols/scale)*(I_RGB.rows/scale);

	
	resize(I_RGB,O_RGB,size);
	static Mat O_GRAY(I_RGB.rows,I_RGB.cols,CV_8UC1,Scalar(0));
	cvtColor(O_RGB,O_GRAY,COLOR_RGB2GRAY);
	

	imshow("Gray Image" , O_GRAY);
	waitKey(30);

	
	static Mat O_HIST(O_GRAY.rows,O_GRAY.cols,CV_8UC1,Scalar(0));

	int numBins=256;
	int bin;
	
	int window_size = 3 ;
	int window_width = O_GRAY.cols/window_size;
	int window_height = O_GRAY.rows/window_size;
	float normFactor = (float)numBins/(window_width*window_height);
	
	
	for(int k=0; k < window_size ; k++){
		for(int l=0;l< window_size;l++){
			int histArray[256]={0};			
			for(int i=k*window_height ; i < (k+1)*window_height ; i++){
				for(int j=l*window_width; j < (l+1)*window_width ; j++){
					bin = O_GRAY.at<uchar>(i,j);             //get grayscale value of pixel
					histArray[bin]=histArray[bin]+1;  //update count of pixels associated with the value
				}
			}

			for(int i=1; i < 256 ; i++){
				histArray[i] = histArray[i-1] + histArray[i];

			}

			float normFactor = (float)numBins/(window_height*window_width);

			for(int i=k*window_height ; i < (k+1)*window_height ; i++){
				for(int j=l*window_width; j < (l+1)*window_width ; j++){
					O_HIST.at<uchar>(i,j) = (int)(histArray[(int)(O_GRAY.at<uchar>(i,j))]*normFactor);
				}
			}
		}
	}

	
	cout << "done " <<endl;

	imwrite("AHE_IMAGE.jpg",O_HIST);
	while(true){
		imshow("Adaptive Histogram Equalized Image",O_HIST);
		waitKey(30);
	}

	return 0;

}
