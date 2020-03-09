#ifndef FUNCTIONS_H
#define FUNCTIONS_H

	void preprocessing(cv::Mat *eye, cv::Mat *thresh_im);

	int compute_threshold(cv::Mat *thresh_im);

	void t_watershed(cv::Mat *thresh_im, std::vector<cv::Point> *m, int step, int BIN_THRESH);

	void show_histogram(std::string const& name, cv::Mat1b const& image, cv::Mat *hist);

	void detect_features(cv::Mat *in, cv::Mat *out, std::vector<cv::Mat> *face_cropped, std::vector<std::vector<cv::Mat>> *eye_cropped, std::vector<cv::Point> *fp, std::vector<std::vector<cv::Point>> *ep);

	void detect_extrema(cv::Mat *gray_eye, std::vector<int> *markers);

#endif	// FUNCTIONS_H