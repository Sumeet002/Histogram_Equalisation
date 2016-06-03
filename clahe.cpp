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
	int clipLimit = 253;
	int outPixCount;
	int window_size = 3;
	int w = O_GRAY.cols/window_size;
	int h = O_GRAY.rows/window_size;
	float normFactor = (float)numBins/(w*h);
	int histArray[36][256]={0};
	
	for(int k=0; k < window_size ; k++){
		for(int l=0;l< window_size;l++){
			outPixCount = 0;			
			for(int i=k*h ; i < (k+1)*h ; i++){
				for(int j=l*w; j < (l+1)*w ; j++){
					bin = (int)O_GRAY.at<uchar>(i,j);
					if(histArray[k*window_size+l][bin] > clipLimit) 
						outPixCount++;
					else            
						histArray[k*window_size+l][bin]+=1;   
				}
			}

			int numDistribute = (int)outPixCount/numBins;

			
			for(int i=0 ; i < 256 ; i++){
				histArray[k*window_size+l][i] += numDistribute; 
			}

			for(int i=1; i < 256 ; i++){
				histArray[k*window_size+l][i] = histArray[k*window_size+l][i-1] + histArray[k*window_size+l][i];

			}
		}
	}

	int ulvalue,uvalue,lvalue,dvalue,rvalue,value,curvalue,newvalue;
	
	for(int k=0; k < window_size ; k+=(window_size-1)){
		for(int l=0 ; l < window_size ; l+=(window_size-1)){
			for(int i = k*h ; i < (k+1)*h ; i++){	
					for(int j=l*w ; j < (l+1)*w; j++){
						curvalue = (int)(O_GRAY.at<uchar>(i,j));
						value = (int)(histArray[k*window_size+l][curvalue]*normFactor);
						O_GRAY.at<uchar>(i,j) = value;
					}
			}
		}
	}		

	for(int l=0 ; l < window_size ; l+=(window_size-1)){
		for(int k=1; k < window_size - 1 ; k++){
	        	for(int i = k*h ; i < (k+1)*h ; i++){	
					for(int j=l*w ; j < (l+1)*w; j++){
				
						curvalue = (int)(O_GRAY.at<uchar>(i,j));
						uvalue = (int)(histArray[(k-1)*window_size+l][curvalue]*normFactor);
						dvalue = (int)(histArray[k*window_size+l][curvalue]*normFactor);
						int A = (uvalue+dvalue)/2;
						O_GRAY.at<uchar>(i,j) = A;
					
					}
				}
			}			
	}

	for(int k=0; k < window_size ; k+=(window_size-1)){
		for(int l=1; l < window_size - 1 ; l++)
				for(int i = k*h ; i < (k+1)*h ; i++){	
					for(int j=l*w ; j < (l+1)*w; j++){
				
					curvalue = (int)(O_GRAY.at<uchar>(i,j));
					rvalue = (int)(histArray[k+l*window_size][curvalue]*normFactor);
					lvalue = (int)(histArray[k+(l-1)*window_size][curvalue]*normFactor);
					int A = (lvalue+rvalue)/2;
					O_GRAY.at<uchar>(i,j) = A;
					
					}
				}
	}

	
	for(int k=1; k < window_size -1 ; k++){
		for(int l=1;l< window_size -1;l++){
			for(int i=k*h ; i < (k+1)*h ; i++){
				for(int j=l*w; j < (l+1)*w ; j++){
					curvalue = (int)(O_GRAY.at<uchar>(i,j));
					
					value = (int)(histArray[k*window_size+l][curvalue]*normFactor);
					uvalue = (int)(histArray[(k-1)*window_size+l][curvalue]*normFactor);
					ulvalue = (int)(histArray[(k-1)*window_size+(l-1)][curvalue]*normFactor);
					lvalue = (int)(histArray[k*window_size+(l-1)][curvalue]*normFactor);

					
					int A = ulvalue*((l*w + w/2) - j)*(i-((k-1)*h + h/2));
					int B = uvalue*(j-((l-1)*w + w/2)) * (i-((k-1)*h+h/2));
					int C = lvalue*((l*w + w/2)-j)*((k*h+h/2)-i);
					int D = value*(j-((l-1)*w + w/2))*((k*h+h/2)-i);

					O_GRAY.at<uchar>(i,j) = (int)((float)(A+B+C+D)/(h*w));	



				}
			}
		}
	}

	
	cout << "done " <<endl;

	imwrite("CLAHE_IMAGE.jpg",O_GRAY);

	while(true){
		imshow("CLAHE Image",O_GRAY);
		waitKey(30);
	}

	return 0;

}
