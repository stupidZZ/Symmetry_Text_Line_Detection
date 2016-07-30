#pragma once
#include "stdafx.h";
#include "CmFile.h";
#include "ICDAR2011DataSet.h"
#include "CommonDataStructure.h"

struct TextlineDetectorParameter
{
	//number of sliding window scales for each octiva
	int scale_num_octiva;	

	//the minimum sliding window scale, it would be 2^min_scale
	int min_scale;	

	//the maximum sliding window scale, it would be 2^max_scale
	int max_scale;	

	//the normalized height of input image;
	int image_normalized_height;	

	//the bin number of lab channel
	int lab_bin;

	//the bin number of gradient channel
	int gradient_bin;

	//the texton_bin of texton channel
	int texton_bin;

	//Only for trainning stage. Pixels whose distance less than this threshold are regard as positive samples;
	double positive_distance_threshold;	

	//Only for trainning stage. Pixels whose distance greater than this threshold are regard as negative samples;
	int negative_distance_threshold;

	//Only for training stage. The maximum positive features in training stage.
	int max_positive_features;

	//Only for training stage. The maximum negative features in training stage.
	int max_negative_features;

	double link_angle_diff_threshold;

	int core_num;

	int shrink;
};

class TextlineDetector
{
	struct ScaleInfo
	{
		int key_down_sampling_scale_num;
		int max_down_sampling_scales;
		int min_down_sampling_scales;
		vector<int> down_sampling_indexs;
		vector<int> real_scales;
		vector<int> down_sampling_scales;
	};

private:
	int train_flag;
	ScaleInfo scale_info;
	TextlineDetectorParameter parameter;

// helper function
private:
	void ShowFeatureMap(const vector<FeatureCollection> &feature, const int width, const int height);

private:
	void InitialzeScaleInfo();
	void ConstructLabMap(const Mat &image, Mat &dst_map);
	void ConstructGradientMap(const Mat &image, Mat &dst_map);
	void ConstructTextonMap(const Mat &image, Mat &dst_map);
	void ExtractLbpHist(const Mat &integral_image, const int scale, FeatureCollection &lbp_hist_feature);
	void ExtractSymmetryFeature(const Mat &integral_image, const int scale, const int bin, FeatureCollection &feature_collection);
	void ExtractFeature(const Mat &image, deque<FeatureCollection> &feature_collections);
	void ExtractFeatureAtScale(const Mat &image, const int scale, FeatureCollection &features);
	void CollectTrainingFeature(const Mat &image, const vector<Rect> &gt_boxes, FeatureCollection &features);
	void SaveFeature(const string &path, FeatureCollection &features);
	void LoadFeature(const string &path, FeatureCollection &features);
	void NmsProb(const Mat &prob_image, Mat &nms_image);
	void SelectSymmetryLinePoint(const Mat &nms_image, Mat &symmetry_line_image, double high_threshold, double low_threshold);
	void LinkSymmetryLine(const Mat &symmetry_line_image, const int scale, vector<vector<cv::Point2d>> &point_group);
	void ResotreRect(const vector<vector<cv::Point2d>> &group_points, int w, int h, double scale_ratio, double window_scale, vector<cv::Rect> &proposals);

public:
	void Train(const ICDAR2011DataSet &dataset);	//Training stage
	void Test(const ICDAR2011DataSet &dataset); //Testing stage
	TextlineDetector(const TextlineDetectorParameter &_parameter);
	TextlineDetector(void);
	~TextlineDetector(void);
};

