#include"iostream"
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>//dtfº¯Êý
#include"Fourior.h"
#include"MyNSp.h"

using namespace std;
using namespace cv;
using namespace MyNSP;

int main()
{
	Mat image = imread("test.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Ori Image", image);
	
	Mat FTImage = FouriorTransit(image);
	//imshow("DFT Image", Newimage);

	Mat ILPFimage = FTImage;
	LowPassFilter(&ILPFimage, 80, IdeaLPF);
	ILPFimage = InvertFouriorTransit(ILPFimage, image.size());
	FouriorTransit(ILPFimage);
	imshow("IdeaLPF Image", ILPFimage);
	waitKey();

	Mat TLPFimage = FTImage;
	LowPassFilter(&TLPFimage, 80, TrapeLPF);
	TLPFimage = InvertFouriorTransit(TLPFimage, image.size());
	FouriorTransit(TLPFimage);
	imshow("TrapeLPF Image", TLPFimage);
	waitKey();

	Mat BLPFimage = FTImage;
	LowPassFilter(&BLPFimage, 80, ButterworthLPF);
	BLPFimage = InvertFouriorTransit(BLPFimage, image.size());
	FouriorTransit(BLPFimage);
	imshow("ButterLPF Image", BLPFimage);
	waitKey();

	Mat ELPFimage = FTImage;
	LowPassFilter(&ELPFimage, 80, ExpLPF);
	ELPFimage = InvertFouriorTransit(ELPFimage, image.size());
	FouriorTransit(ELPFimage);
	imshow("ExpLPF Image", ELPFimage);
	waitKey();

	return 0;
}