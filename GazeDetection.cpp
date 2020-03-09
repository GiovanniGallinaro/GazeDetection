#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>

#include "functions.h"

using namespace std;
using namespace cv;

const string DIRECTORY = "images_T2S/";
const string FORMAT = ".jpg";
const int INPUT_SIZE = 16;

int main()
{
	//************** INPUT READING **************
	Mat inputs[INPUT_SIZE];
	for (int i = 0; i < INPUT_SIZE; i++) 
	{
		Mat temp = imread(DIRECTORY + to_string(i + 1) + FORMAT);

		// check if input is valid
		if (temp.cols > 0 && temp.rows > 0) 
		{
			inputs[i] = temp;
		}
	}

	// vector of indexes of the selected images for the demo
	vector<int> im_index = { 3, 4, 5, 6, 10, 12, 14 };

	// iterate images
	for (int i = 0; i < im_index.size(); i++)
	{
		Mat in = inputs[im_index[i]];

		//************** FEATURE DETECTION **************
		// detect and draw features
		Mat out = in.clone();			// will have rectangles drawn
		vector<Mat> face_cropped;			// vector of faces
		vector<vector<Mat>> eye_cropped;	// vector of vector for the eyes
		vector<Point> fp;		// coordinates of the rectangles
		vector<vector<Point>> ep;
		detect_features(&in, &out, &face_cropped, &eye_cropped, &fp, &ep);

		// show detected features
		namedWindow("Detected features", WINDOW_NORMAL);
		imshow("Detected features", out);
		//imwrite(DIRECTORY + to_string(i) + "_features.jpg", out);

		Mat final_out = in.clone();
		// iterate faces
		for (int j = 0; j < eye_cropped.size(); j++)
		{
			
			vector<float> markers;

			// iterate eyes
			for (int k = 0; k < eye_cropped[j].size(); k++)
			{
				//************** PRE PROCESSING **************

				// extract cropped eye
				Mat eye = eye_cropped[j][k];

				Mat thresh_im;
				preprocessing(&eye, &thresh_im);

				// quantize the obtained value
				int BIN_THRESH = compute_threshold(&thresh_im);

				//************** SEGMENT WITH MULTIPLE THRESHOLDS **************
				int step = 15;
				vector<Point> points;
				t_watershed(&thresh_im, &points, step, BIN_THRESH);

				cout << "avg1: " + to_string(points[1].x) + ", " + to_string(points[1].y) << endl;
				cout << "avg2: " + to_string(points[2].x) + ", " + to_string(points[2].y) << endl;

				// draw the results on the original image
				drawMarker(final_out, Point(fp[j].x + ep[j][k].x + points[0].x, fp[j].y + ep[j][k].y + eye_cropped[j][k].rows / 4 + points[0].y), Scalar(0, 0, 255), MARKER_CROSS);
				circle(final_out, Point(fp[j].x + ep[j][k].x + points[1].x, fp[j].y + ep[j][k].y + eye_cropped[j][k].rows / 4 + points[1].y), 2, Scalar(0, 255, 0), -1);
				circle(final_out, Point(fp[j].x + ep[j][k].x + points[2].x, fp[j].y + ep[j][k].y + eye_cropped[j][k].rows / 4 + points[2].y), 2, Scalar(0, 255, 0), -1);

				// compute distances between the found keypoints
				int width = points[2].x - points[1].x;		// distance between the two sclera corners
				int pupil = points[0].x - points[1].x;		// distance between the iris and the left corner
				float rate;

				// report error if iris distance is not valid
				if (pupil < 0) {
					rate = 0;
					cout << "ERROR" << endl;
				}

				// compute percentage rate of the iris distance
				else 
				{
					rate = (float)pupil / (float)width;
				}
				markers.push_back(rate);
			}

			// if both eyes are found
			if (markers.size() >= 2)
			{
				cout << "rate 1: " + to_string(markers[markers.size() - 2]) << endl;
				cout << "rate 2: " + to_string(markers[markers.size() - 1]) << endl;

				// determine gaze direction
				if (markers[markers.size() - 2] > 0.54 || markers[markers.size() - 1] > 0.54)
				{
					cout << "FACE " + to_string(j) + ": LEFT" << endl;
				}
				else if (markers[markers.size() - 2] < 0.46 || markers[markers.size() - 1] < 0.46)
				{
					cout << "FACE " + to_string(j) + ": RIGHT" << endl;
				}
				else
				{
					cout << "FACE " + to_string(j) + ": STRAIGHT" << endl;
				}
			}

			// if just one eye is found
			else if (markers.size() > 0 && markers.size() < 2)
			{
				cout << "rate 1: " + to_string(markers[markers.size() - 1]) << endl;

				if (markers[markers.size() - 1] > 0.55)
				{
					cout << "FACE " + to_string(j) + ": LEFT" << endl;
				}
				else if (markers[markers.size() - 1] < 0.45)
				{
					cout << "FACE " + to_string(j) + ": RIGHT" << endl;
				}
				else
				{
					cout << "FACE " + to_string(j) + ": STRAIGHT" << endl;
				}
			}

			namedWindow("Final Result " + to_string(i), WINDOW_NORMAL);
			imshow("Final Result " + to_string(i), final_out);
			imwrite(DIRECTORY + "output" + to_string(i) + FORMAT, final_out);
			waitKey(0);
		}
	}

	waitKey(0);
	destroyAllWindows();

	return 0;
}