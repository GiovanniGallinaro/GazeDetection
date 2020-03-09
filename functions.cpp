#include <iostream>
#include <opencv2/opencv.hpp>

#include "functions.h"

using namespace std;
using namespace cv;

void preprocessing(Mat *eye, Mat *thresh_im)
{
	// crop the image
	*eye = (*eye)(Rect(0, (*eye).cols / 4, (*eye).rows, (*eye).cols / 2));

	// convert to gray
	Mat gray_eye;
	cvtColor(*eye, gray_eye, COLOR_BGR2GRAY);

	// apply contrast
	gray_eye.convertTo(*thresh_im, -1, 1.5, 30);
	//equalizeHist(thresh_im, thresh_im);
}

int compute_threshold(Mat *thresh_im)
{
	// set histogram bins count
	int bins = 256;
	int histSize[] = { bins };

	// set ranges for histogram bins
	float lranges[] = { 0, 256 };
	const float* ranges[] = { lranges };

	// create matrix for histogram
	int channels[] = { 0 };
	Mat hist;

	// initialize value corresponding to the amount of light pixels
	int white = 0;

	// calculate the histogram of the image
	calcHist(thresh_im, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);

	// sum all the values of the histogram multiplied by the weight h (= intensity value)
	for (int h = 0; h < hist.rows; h++) {
		white += (int)hist.at<float>(h, 1) * h;
	}

	// normalize wrt to the size of the image
	white = white / ((*thresh_im).rows*(*thresh_im).cols);

	// return the equantize value
	return floor(white / 30) * 30;
}

void t_watershed(Mat *thresh_im, vector<Point> *m, int step, int BIN_THRESH)
{
	// initialize the coordinates of the iris and sclera corners
	int avg_c_x = 0, avg_c_y = 0, avg_min_x = 0, avg_min_y = 0, avg_max_x = 0, avg_max_y = 0;

	// initialize values for the thresholds selection
	int min_thresh = BIN_THRESH - 3 * step;		// minimum value for the threshold
	int max_thresh = BIN_THRESH + 2 * step;		// maximum value for the threshold
	int range = (max_thresh - min_thresh) / step;		// range between which the image is segmented

	// iterate over the possible thresholds
	for (int r = min_thresh; r < max_thresh; r += step)
	{
		// apply the threshold
		Mat t;
		threshold(*thresh_im, t, r, 255, THRESH_BINARY);

		// compute inverse image
		bitwise_not(t, t);

		// find centroids of the black cluster
		Moments m = moments(t, true);

		// save the coordinates of the found centroid
		avg_c_x += m.m10 / m.m00;
		avg_c_y += m.m01 / m.m00;

		// compute inverse image to return to the original
		bitwise_not(t, t);

		// detect the coordinates of the sclera corners and store them into a vector
		vector<int> temp_mark;
		detect_extrema(&t, &temp_mark);
		cout << "markers min: " + to_string(temp_mark[0]) + ", " + to_string(temp_mark[1]) << endl;
		cout << "markers max: " + to_string(temp_mark[2]) + ", " + to_string(temp_mark[3]) << endl;
		avg_min_x += temp_mark[0];
		avg_min_y += temp_mark[1];
		avg_max_x += temp_mark[2];
		avg_max_y += temp_mark[3];
	}

	// compute the average values
	avg_c_x = avg_c_x / range;
	avg_c_y = avg_c_y / range;
	Point c(avg_c_x, avg_c_y);
	(*m).push_back(c);
	avg_min_x = avg_min_x / range;
	avg_min_y = avg_min_y / range;
	Point l(avg_min_x, avg_min_y);
	(*m).push_back(l);
	avg_max_x = avg_max_x / range;
	avg_max_y = avg_max_y / range;
	Point r(avg_max_x, avg_max_y);
	(*m).push_back(r);
}

void show_histogram(std::string const& name, cv::Mat1b const& image, Mat *hist)
{
	// Set histogram bins count
	int bins = 256;
	int histSize[] = { bins };
	// Set ranges for histogram bins
	float lranges[] = { 0, 256 };
	const float* ranges[] = { lranges };
	// create matrix for histogram
	//cv::Mat hist;
	int channels[] = { 0 };

	// create matrix for histogram visualization
	int const hist_height = 256;
	cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);

	cv::calcHist(&image, 1, channels, cv::Mat(), *hist, 1, histSize, ranges, true, false);

	double max_val = 0;
	minMaxLoc(*hist, 0, &max_val);

	// visualize each bin
	for (int b = 0; b < bins; b++) {
		float const binVal = (*hist).at<float>(b);
		int   const height = cvRound(binVal*hist_height / max_val);
		cv::line
		(hist_image
			, cv::Point(b, hist_height - height), cv::Point(b, hist_height)
			, cv::Scalar::all(255)
		);
	}
	cv::imshow(name, hist_image);
}

void detect_features(Mat *in, Mat *out, vector<Mat> *face_cropped, vector<vector<Mat>> *eye_cropped, vector<Point> *fp, vector<vector<Point>> *ep)
{
	// initialize and load classifiers for face and eyes
	CascadeClassifier face_cascade;
	face_cascade.load("haarcascade_frontalface_alt.xml");
	CascadeClassifier eye_cascade;
	eye_cascade.load("haarcascade_eye.xml");

	// detect faces
	vector<Rect> faces;
	face_cascade.detectMultiScale(*in, faces);

	// initialize vector of vector for the eyes
	vector<vector<Rect>> eyes;

	// iterate faces
	for (int i = 0; i < faces.size(); i++)
	{
		// draw rectangle around detected face
		rectangle(*out, faces[i].tl(), faces[i].br(), Scalar(0, 0, 255), 4);

		Mat temp_face = (*in)(faces[i]);		// crop face
		(*face_cropped).push_back(temp_face);		// update faces vector
		(*fp).push_back(faces[i].tl());			// update faces positions vector

		vector<Mat> temp_eye_vector;
		(*eye_cropped).push_back(temp_eye_vector);
		vector<Point> temp_ep_vector;
		(*ep).push_back(temp_ep_vector);
		
		// detect eyes
		vector<Rect> temp_eyes;
		eye_cascade.detectMultiScale(temp_face, temp_eyes);

		// update eyes vector
		eyes.push_back(temp_eyes);

		// iterate eyes
		for (int j = 0; j < eyes[i].size(); j++)
		{
			// draw rectangle around detected eye
			rectangle(*out, faces[i].tl() + eyes[i][j].tl(), faces[i].tl() + eyes[i][j].br(), Scalar(0, 0, 255), 2);

			(*eye_cropped)[i].push_back(temp_face(eyes[i][j]));		// update eyes vector
			(*ep)[i].push_back(eyes[i][j].tl());		// update eyes positions vector
		}
	}
}

void detect_extrema(Mat *gray_eye, vector<int> *markers)
{
	// initialize min and max values
	int max_x = 1;
	int min_x = (*gray_eye).cols;

	// y values corresponding to the points in which the x coordinates are max/min
	int max_y = 1;
	int min_y = 1;

	// store y (case in which multiple points with max/min x are found)
	vector<int> maxs_y, mins_y;

	// iterate each pixel of the image
	for (int x = 1; x < (*gray_eye).cols - 1; x++) {
		for (int y = 1; y < (*gray_eye).rows - 1; y++) {
			// take min and max points from the black cluster
			if ((int)(*gray_eye).at<uchar>(y, x) < 150) {
				if (x >= max_x) {
					max_x = x;
					maxs_y.push_back(y);
				}
				if (x <= min_x) {
					min_x = x;
					mins_y.push_back(y);
				}
			}
		}
	}

	//take average values
	int sum = 0;
	for (int i = 0; i < maxs_y.size(); i++)
	{
		sum += maxs_y[i];
	}
	if (maxs_y.size() > 0)
	{
		max_y = sum / maxs_y.size();
	}
	int sum2 = 0;
	for (int i = 0; i < mins_y.size(); i++)
	{
		sum2 += mins_y[i];
	}
	if (mins_y.size() > 0)
	{
		min_y = sum2 / mins_y.size();
	}

	// update vectors
	(*markers).push_back(min_x);
	(*markers).push_back(min_y);
	(*markers).push_back(max_x);
	(*markers).push_back(max_y);
}