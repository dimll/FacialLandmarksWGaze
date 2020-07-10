#include<dlib/image_processing/frontal_face_detector.h>
#include<dlib/image_processing.h>
#include<dlib/opencv.h>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>

#define EYE_EAR_THRES 0.2


using namespace cv;
using namespace std;
using namespace dlib;

const std::string detectorDir = "D:/Files/ece/Dev/DlibFull/shape_predictor_68_face_landmarks.dat";


//Function Headers
void detectLandmarks(Mat& frame, frontal_face_detector faceDetector, shape_predictor landmarkDetector,
	std::vector<dlib::rectangle>& faces, float resizeScale, int skipFrames, int frameCounter);

void drawPolyline(cv::Mat& image, full_object_detection landmarks, int start, int end, bool isClosed = false);
void drawPolylines(cv::Mat& image, full_object_detection landmarks);
void displayPoints(cv::Mat& image);


//Test
void isolate(Mat frame, std::vector<cv::Point> eyePoints, Mat* eyeFrame, cv::Point* origin, cv::Point* center);
int calculateBestThreshold(Mat eyeFrame);
cv::Mat processEye(Mat eyeFrame, int threshold);
float calculateIrisSize(Mat irisFrame);
void detectIris(Mat eyeFrame, int threshold, int* x, int* y);
bool isBlinking(std::vector<cv::Point> leftEyePoints, std::vector<cv::Point> rightEyePoints);
bool isBlinkingv2(std::vector<cv::Point> leftEyePoints);
double eyeAspectRatio(std::vector<cv::Point> eyePoints);
cv::Point2f middlePoint(cv::Point2f p1, cv::Point2f p2);
cv::Point2f middlePoint(cv::Point p1, cv::Point p2);

cv::Point detectGaze(Mat frame, std::vector<cv::Point> eyePoints);


int eyeGazeThreshold = 0;

std::vector<cv::Point2f> faceLandmarkPts;
std::vector<cv::Point> rightEyePoints, leftEyePoints;
cv::Point rightGaze, leftGaze;
bool faceDetected = false;

char key;


int main() {

	bool firstFrame = true;
	double faceSize;

	float eyeBrowsDif, eyeBrowsInit, eyeBrowsCurrent;
	float rightGazeX, rightGazeXInit;
	float mouthOpen, mouthWide, mouthInitWide;



	//Open the web camera
	VideoCapture videoCapture(0);

	if (!videoCapture.isOpened()) {
		cout << "Web camera can't be opened" << endl;
		return -1;
	}
	//Set the video quality
	if (videoCapture.set(CAP_PROP_FRAME_WIDTH, 1280) && videoCapture.set(CAP_PROP_FRAME_HEIGHT, 720))
	{
		cout << "Video quality set to: " << 1280 << " x " << 720 << endl;
	}
		

	//define the face detector
	frontal_face_detector faceDetector = get_frontal_face_detector();
	//define landmark detector
	shape_predictor landmarkDetector;
	//load face landmark model
	deserialize(detectorDir) >> landmarkDetector;

	//define resize height
	float resizeHeight = 480;
	

	//define skip frames
	int skipFrames = 5;

	//Get first frame
	Mat frame;
	videoCapture.read(frame);
	

	//calculate resize scale
	float height = frame.rows;
	float resizeScale = height / resizeHeight;
	int frameCounter = 0;
	//define to hold detected faces
	std::vector<dlib::rectangle> faces;

	//cv::namedWindow("Original", WINDOW_AUTOSIZE);
	cv::namedWindow("Face Landmarks", WINDOW_AUTOSIZE);

	while (1)
	{
		

		if (videoCapture.read(frame))
		{

			
			detectLandmarks(frame, faceDetector, landmarkDetector, faces, resizeScale, skipFrames, frameCounter);
			imshow("Face Landmarks", frame);

			//Press ESC or q to exit the program
			key = waitKey(1);
			if (key == 27 || key == 113)
			{
				break;
			}

			//increment frame counter
			frameCounter++;

			if (frameCounter == 100) {
				frameCounter = 0;
			}
			
		}
	}
	
	videoCapture.release();
	cv::destroyAllWindows();

	return 0;
}

void detectLandmarks(Mat& frame, frontal_face_detector faceDetector, shape_predictor landmarkDetector,
	std::vector<dlib::rectangle>& faces, float resizeScale, int skipFrames, int frameCounter)
{
	//Clear the points of the previous frame
	faceLandmarkPts.clear();
	rightEyePoints.clear();
	leftEyePoints.clear();

	//to store resized image
	Mat smallFrame;

	//resize frame to smaller image
	resize(frame, smallFrame, Size(), 1.0 / resizeScale, 1.0 / resizeScale);

	//change to dlib image format
	cv_image<bgr_pixel> dlibImageSmall(smallFrame);
	cv_image<bgr_pixel> dlibImage(frame);

	//detect faces at interval of skipFrames
	if (frameCounter % skipFrames == 0) {
		faces = faceDetector(dlibImageSmall);
	}

	if (faces.size() == 0)
	{
		cout << "Can't detect face in this frame" << endl;
		faceDetected = false;
		return;
	}
	faceDetected = true;

	//loop over faces
	for (int i = 0; i < faces.size(); i++) {

		//scale the rectangle coordinates as we did face detection on resized smaller image
		dlib::rectangle rect(int(faces[i].left() * resizeScale),
			int(faces[i].top() * resizeScale),
			int(faces[i].right() * resizeScale),
			int(faces[i].bottom() * resizeScale));

		//Face landmark detection
		full_object_detection faceLandmarks = landmarkDetector(dlibImage, rect);

		for (int i = 0; i < 68; i++)
		{
			cv::Point2d p = cv::Point2d(faceLandmarks.part(i).x(), faceLandmarks.part(i).y());
			faceLandmarkPts.push_back(p);
			if (i >= 36 && i <= 41) rightEyePoints.push_back(p);
			if (i >= 42 && i <= 47) leftEyePoints.push_back(p);
		}

		if (isBlinking(leftEyePoints, rightEyePoints)) cout << "Blinking" << endl;
		//if (isBlinkingv2(leftEyePoints)) cout << "Blinking left" << endl;

		rightGaze = detectGaze(frame, rightEyePoints);
		leftGaze = detectGaze(frame, leftEyePoints);

		//drawPolylines(frame, faceLandmark);
		displayPoints(frame);

	}
}

void drawPolyline(cv::Mat& image, full_object_detection landmarks, int start, int end, bool isClosed)
{
	std::vector<cv::Point> points;
	for (int i = start; i <= end; i++) {
		points.push_back(cv::Point(landmarks.part(i).x(), landmarks.part(i).y()));

	}
	cv::polylines(image, points, isClosed, cv::Scalar(0, 255, 255), 2, 16);
}

void drawPolylines(cv::Mat& image, full_object_detection landmarks) 
{
	drawPolyline(image, landmarks, 0, 16);              //jaw line
	drawPolyline(image, landmarks, 17, 21);             //left eyebrow
	drawPolyline(image, landmarks, 22, 26);             //right eyebrow
	drawPolyline(image, landmarks, 27, 30);             //Nose bridge
	drawPolyline(image, landmarks, 30, 35, true);       //lower nose
	drawPolyline(image, landmarks, 36, 41, true);       //left eye
	drawPolyline(image, landmarks, 42, 47, true);       //right eye
	drawPolyline(image, landmarks, 48, 59, true);       //outer lip
	drawPolyline(image, landmarks, 60, 67, true);       //inner lip
}

void displayPoints(cv::Mat& image)
{
	for (int i = 0; i < faceLandmarkPts.size(); i++)
	{
		cv::circle(image, faceLandmarkPts[i], 1, cv::Scalar(0, 0, 255), FILLED);
		//if (i == 48 || i == 54) cv::circle(image, faceLandmarkPts[i], 1, cv::Scalar(0, 0, 255), FILLED);
	}
	cv::circle(image, rightGaze, 2, cv::Scalar(0, 255, 0), FILLED);
	cv::circle(image, leftGaze, 2, cv::Scalar(0, 255, 0), FILLED);
}


cv::Point detectGaze(Mat frame, std::vector<cv::Point> eyePoints)
{

	Mat eyeFrame;
	cv::Point origin, center;
	int thres, x, y;

	cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

	isolate(frame, eyePoints, &eyeFrame, &origin, &center);

	/*cout << "Eye Frame size = " << eyeFrame.size() << endl;
	cout << "Center X = " << center.x << endl;
	cout << "Center Y = " << center.x << endl;*/

	thres = calculateBestThreshold(eyeFrame);

	detectIris(eyeFrame, thres, &x, &y);
	return cv::Point(origin.x + x, origin.y + y);

}

//Isolates the eye in a diffrent frame
void isolate(Mat frame, std::vector<cv::Point> eyePoints, Mat* eyeFrame, cv::Point* origin, cv::Point* center)
{
	Mat blackFrame, mask, eye;
	std::vector<std::vector<cv::Point> > eyePointArr;
	int height, width;

	int margin = 5;
	int minX = eyePoints[0].x;
	int maxX = eyePoints[0].x;
	int minY = eyePoints[0].y;
	int maxY = eyePoints[0].y;

	blackFrame = Mat::zeros(frame.rows, frame.cols, 0);


	mask = Mat::Mat(frame.rows, frame.cols, 0, 255);
	eyePointArr.push_back(eyePoints);
	fillPoly(mask, eyePointArr, (0, 0, 0));


	eye = frame.clone();
	cv::bitwise_not(blackFrame, eye, mask);


	for (int i = 1; i < eyePoints.size(); i++)
	{
		if (eyePoints[i].x < minX) minX = eyePoints[i].x;
		if (eyePoints[i].x > maxX) maxX = eyePoints[i].x;
		if (eyePoints[i].y < minY) minY = eyePoints[i].y;
		if (eyePoints[i].y > maxY) maxY = eyePoints[i].y;
	}

	minX -= margin;
	maxX += margin;
	minY -= margin;
	maxY += margin;

	*eyeFrame = eye(cv::Range(minY, maxY), cv::Range(minX, maxX));
	height = eyeFrame->rows;
	width = eyeFrame->cols;
	*origin = cv::Point(minX, minY);
	*center = cv::Point(width / 2, height / 2);

}


//Calculates the best threshold for the gaze detection 
//from an eye frame
int calculateBestThreshold(Mat eyeFrame)
{
	Mat irisFrame;
	float averageIrisSize = 0.48;
	int bestThreshold;
	float minValue = 1000;
	float irisSize, temp;

	for (int threshold = 5; threshold <= 100; threshold += 5)
	{
		irisFrame = processEye(eyeFrame, threshold);
		irisSize = calculateIrisSize(irisFrame);

		temp = std::abs(irisSize - averageIrisSize);
		if (temp < minValue)
		{
			bestThreshold = threshold;
			minValue = temp;
		}
	}
	return bestThreshold;

}

//Blurs, erodes and binarizes the eye frame 
cv::Mat processEye(Mat eyeFrame, int threshold)
{
	cv::Mat newFrame, kernel;



	kernel = Mat::Mat(3, 3, 0, 1);
	cv::bilateralFilter(eyeFrame, newFrame, 10, 15, 15);

	/*namedWindow("Blur", cv::WINDOW_NORMAL);
	imshow("Blur", eyeFrame);*/

	cv::erode(newFrame, newFrame, kernel, cv::Point(-1, -1), 3);

	//namedWindow("Erode", cv::WINDOW_NORMAL);
	//imshow("Erode", newFrame);

	cv::threshold(newFrame, newFrame, threshold, 255, cv::THRESH_BINARY);

	//namedWindow("Binarized", cv::WINDOW_NORMAL);
	//imshow("Binarized", newFrame);


	return newFrame;

}

//Returns the percentage of space that the iris
//takes up on the surface of the eye
float calculateIrisSize(Mat irisFrame)
{
	int height, width, nbPixels, nbBlacks;

	//irisFrame = irisFrame(cv::Range(5, -5), cv::Range(5, -5));
	irisFrame = irisFrame(cv::Range(5, irisFrame.rows - 6), cv::Range(5, irisFrame.cols - 6));
	height = irisFrame.rows;
	width = irisFrame.cols;
	nbPixels = height * width;
	nbBlacks = nbPixels - cv::countNonZero(irisFrame);

	return ((float)nbBlacks / (float)nbPixels);

}

//Detects the iris
void detectIris(Mat eyeFrame, int threshold, int* x, int* y)
{
	Mat irisFrame;
	std::vector<std::vector<cv::Point>> contours;
	std::vector<std::vector<cv::Point>> contoursv2;
	std::vector<cv::Point> minContour;
	cv::Moments mnts;


	irisFrame = processEye(eyeFrame, threshold);

	//namedWindow("Iris Frame", cv::WINDOW_AUTOSIZE);
	//imshow("Iris Frame", irisFrame);

	cv::findContours(irisFrame, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

	if (contours.size() >= 2)
	{
		//Get the last 2 contours
		contoursv2.push_back(contours.back());
		contours.pop_back();
		contoursv2.push_back(contours.back());

		if ((cv::contourArea(contoursv2[0])) > (cv::contourArea(contoursv2[1]))) minContour = contoursv2[1];
		else minContour = contoursv2[0];

		try
		{
			mnts = cv::moments(minContour);
			*x = (int)(mnts.m10 / mnts.m00);
			*y = (int)(mnts.m01 / mnts.m00);

		}
		catch (std::exception & e)
		{
			cout << e.what() << endl;
		}
	}

}

//Returns true is the user is blinking else returns false
bool isBlinking(std::vector<cv::Point> leftEyePoints, std::vector<cv::Point> rightEyePoints)
{
	double leftEAR = eyeAspectRatio(leftEyePoints);
	double rightEAR = eyeAspectRatio(rightEyePoints);
	double ear = (leftEAR + rightEAR) / 2;

	//cout << "EAR = " << ear << endl;
	if (ear <= EYE_EAR_THRES) return true;
	return false;
}

bool isBlinkingv2(std::vector<cv::Point> eyePoints)
{
	cout << "EAR = " << eyeAspectRatio(eyePoints) << endl;
	if (eyeAspectRatio(eyePoints) <= EYE_EAR_THRES) return true;
	return false;
}


//Calculates the Eye Aspect Ratio based on "Real-Time Eye Blink Detection using Facial Landmarks"
//paper by Soukupova and Cech 
double eyeAspectRatio(std::vector<cv::Point> eyePoints)
{
	double a, b, c, ear;
	a = hypot((eyePoints[1].x - eyePoints[5].x), (eyePoints[1].y - eyePoints[5].y));
	b = hypot((eyePoints[2].x - eyePoints[4].x), (eyePoints[2].y - eyePoints[4].y));
	c = hypot((eyePoints[0].x - eyePoints[3].x), (eyePoints[0].y - eyePoints[3].y));

	try {
		ear = (a + b) / (2 * c);
		return ear;
	}
	catch (std::exception & e) {
		return NULL;
	}

}

//Calculates the middle point between two OpenCV points
cv::Point2f middlePoint(cv::Point2f p1, cv::Point2f p2)
{
	cv::Point2f middlePt;
	middlePt.x = (p1.x + p2.x) / 2;
	middlePt.y = (p1.y + p2.y) / 2;
	return middlePt;
}

cv::Point2f middlePoint(cv::Point p1, cv::Point p2)
{
	cv::Point2f middlePt;
	middlePt.x = (p1.x + p2.x) / 2;
	middlePt.y = (p1.y + p2.y) / 2;
	return middlePt;
}





