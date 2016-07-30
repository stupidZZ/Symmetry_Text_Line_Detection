#include "stdafx.h"
#include "TextlineDetector.h"

void TextlineDetector::ExtractFeature(const Mat &image, deque<FeatureCollection> &feature_collections)
{
	//Collect down sampling images
	vector<int> pad_margins;
	vector<Mat> down_sampling_images;
	Mat cur_image = image;
	
	for(int idx = 0; idx < scale_info.key_down_sampling_scale_num; idx++)
	{
		// find max pad margin;
		int pad_margin = 0;
		for(int scale_idx = 0; scale_idx < scale_info.down_sampling_scales.size(); scale_idx++)
		{
			if(scale_info.down_sampling_indexs[scale_idx] == idx)
			{
				pad_margin = max(pad_margin, scale_info.down_sampling_scales[scale_idx]);
			}
		}
		pad_margin = pad_margin * 2 + 5;
		//pad image
		Mat image_pad;
		copyMakeBorder(cur_image, image_pad, pad_margin, pad_margin, pad_margin, pad_margin, BORDER_REPLICATE);
		down_sampling_images.push_back(image_pad);
		pad_margins.push_back(pad_margin);
		pyrDown(cur_image, cur_image, Size(cur_image.cols/2, cur_image.rows/2));
	}

	// extract features 
	int cur_scale_index = -1;
	vector<int> bins;
	vector<vector<Mat>> symmetry_integral_images_set;
	vector<Mat> lbp_integral_images_set;

	bins.push_back(parameter.lab_bin);
	bins.push_back(parameter.lab_bin);
	bins.push_back(parameter.lab_bin);
	bins.push_back(parameter.gradient_bin);
	bins.push_back(parameter.texton_bin);

	//pre compute integral image
	int max_down_sampling_indexs = 0;
	for(int idx = 0; idx < scale_info.down_sampling_scales.size(); idx++)
	{
		max_down_sampling_indexs = max(max_down_sampling_indexs, scale_info.down_sampling_indexs[idx]);
	}
	max_down_sampling_indexs++;
	symmetry_integral_images_set.resize(max_down_sampling_indexs);
	lbp_integral_images_set.resize(max_down_sampling_indexs);
	
	#pragma omp parallel for
	for(int scale_idx = 0; scale_idx < max_down_sampling_indexs; scale_idx++)
	{
		Mat lab_map;
		Mat grad_map;
		Mat texton_map;
		vector<Mat> integral_image_loop;
		Mat integral_image;
		symmetry_integral_images_set[scale_idx].clear();

		// Construct feature maps
		ConstructLabMap(down_sampling_images[scale_idx], lab_map);
		ConstructGradientMap(down_sampling_images[scale_idx], grad_map);
		//about 3s
		ConstructTextonMap(down_sampling_images[scale_idx], texton_map);

		// split and merge feature maps
		vector<Mat> feature_maps;
		split(lab_map, feature_maps);
		feature_maps.push_back(grad_map);
		feature_maps.push_back(texton_map);

		symmetry_integral_images_set[scale_idx].resize(feature_maps.size());
		for(int channel_idx = 0; channel_idx < feature_maps.size(); channel_idx++)
		{
			FeatureCollection top_middle, bottom_middle, middle_middle;
			// Calculate integral images for each code 
			integral_image_loop.clear();
			for(int code = 1; code <= bins[channel_idx]; code++)
			{
				integral((feature_maps[channel_idx] == code) / 255, integral_image);
				integral_image_loop.push_back(integral_image(Rect(1, 1, integral_image.cols - 1, integral_image.rows - 1)).clone());
			}

			Mat integral_image_merged;
			cv::merge(integral_image_loop, integral_image_merged);
			symmetry_integral_images_set[scale_idx][channel_idx] = integral_image_merged.clone();
		}

		integral_image_loop.resize(parameter.texton_bin);
		for(int code = 1; code <= parameter.texton_bin; code++)
		{
			integral((texton_map == code)/255, integral_image);
				
			integral_image_loop[code - 1] = integral_image(Rect(1, 1, integral_image.cols - 1, integral_image.rows - 1)).clone();
		}
		cv::merge(integral_image_loop, lbp_integral_images_set[scale_idx]);
	}


	feature_collections.resize(scale_info.down_sampling_scales.size());
	for(int idx = 0; idx < scale_info.down_sampling_scales.size(); idx++)
	{
		int cur_scale_index = scale_info.down_sampling_indexs[idx];
		vector<Mat> &symmetry_integral_images = symmetry_integral_images_set[cur_scale_index];
		Mat &lbp_integral_images = lbp_integral_images_set[cur_scale_index];
#if 0
		//compute integral image
		while(cur_scale_index != scale_info.down_sampling_indexs[idx])
		{
			Mat integral_image;
			symmetry_integral_images.clear();
			cur_scale_index++;

			// Construct feature maps
			ConstructLabMap(down_sampling_images[cur_scale_index], lab_map);
			ConstructGradientMap(down_sampling_images[cur_scale_index], grad_map);
			//about 3s
			ConstructTextonMap(down_sampling_images[cur_scale_index], texton_map);

			// split and merge feature maps
			vector<Mat> feature_maps;
			split(lab_map, feature_maps);
			feature_maps.push_back(grad_map);
			feature_maps.push_back(texton_map);

			for(int channel_idx = 0; channel_idx < feature_maps.size(); channel_idx++)
			{
				FeatureCollection top_middle, bottom_middle, middle_middle;
				// Calculate integral images for each code 
				integral_image_loop.clear();
				for(int code = 1; code <= bins[channel_idx]; code++)
				{
					integral((feature_maps[channel_idx] == code) / 255, integral_image);
					integral_image_loop.push_back(integral_image(Rect(1, 1, integral_image.cols - 1, integral_image.rows - 1)).clone());
				}

				Mat integral_image_merged;
				merge(integral_image_loop, integral_image_merged);
				symmetry_integral_images.push_back(integral_image_merged.clone());
			}

			integral_image_loop.resize(parameter.texton_bin);
			for(int code = 1; code <= parameter.texton_bin; code++)
			{
				integral((texton_map == code)/255, integral_image);
				
				integral_image_loop[code - 1] = integral_image(Rect(1, 1, integral_image.cols - 1, integral_image.rows - 1)).clone();
			}
			merge(integral_image_loop, lbp_integral_images);
		}		
#endif
		// Obtain feature maps information
		int margin = scale_info.down_sampling_scales[idx] * 2 + 1;
		int feature_count = (down_sampling_images[cur_scale_index].rows - 2 * margin + 1) * (down_sampling_images[cur_scale_index].cols - 2 * margin + 1);
		vector<FeatureCollection> symmetry_feature_collections;
		symmetry_feature_collections.resize(symmetry_integral_images.size());

		#pragma omp parallel for
		for(int channel_idx = 0; channel_idx < symmetry_integral_images.size(); channel_idx++)
		{
			// Initialize feature maps;
			FeatureCollection feature_collection;
			feature_collection.resize(feature_count);
			for(int feature_idx = 0; feature_idx < feature_count; feature_idx++)
			{
				feature_collection[feature_idx].featureArray.resize(3);
			}

			//extract symmetry feature
			ExtractSymmetryFeature(symmetry_integral_images[channel_idx], scale_info.down_sampling_scales[idx], bins[channel_idx], feature_collection);
			symmetry_feature_collections[channel_idx] = feature_collection;
		}
#if 1
		// Calculate appearance feature
		FeatureCollection lbp_hist_feature;
		lbp_hist_feature.resize(feature_count);

		#pragma omp parallel for
		for(int feature_idx = 0; feature_idx < feature_count; feature_idx++)
		{
			lbp_hist_feature[feature_idx].featureArray.resize(parameter.texton_bin);
		}

		ExtractLbpHist(lbp_integral_images, scale_info.down_sampling_scales[idx], lbp_hist_feature);
		
		// Merge features
		FeatureCollection feature_collection_at_scale;
		int used_feature_cnt = 0;

		vector<int> used_feature_cnt_map;
		used_feature_cnt_map.resize(lbp_hist_feature.size());
		for(int feature_idx = 0; feature_idx < lbp_hist_feature.size(); feature_idx++)
		{
			//exclude pad element
			if((lbp_hist_feature[feature_idx].x < pad_margins[cur_scale_index]) || (lbp_hist_feature[feature_idx].x >= down_sampling_images[cur_scale_index].cols - pad_margins[cur_scale_index]) 
				||(lbp_hist_feature[feature_idx].y < pad_margins[cur_scale_index]) || (lbp_hist_feature[feature_idx].y >= down_sampling_images[cur_scale_index].rows - pad_margins[cur_scale_index]))
			{
				used_feature_cnt_map[feature_idx] = -1;
			}
			else
			{
				used_feature_cnt_map[feature_idx] = used_feature_cnt++;
			}
		}
		feature_collection_at_scale.resize(used_feature_cnt);


		#pragma omp parallel for
		for(int feature_idx = 0; feature_idx < lbp_hist_feature.size(); feature_idx++)
		{
			//exclude pad element
			/*if((lbp_hist_feature[feature_idx].x < pad_margins[cur_scale_index]) || (lbp_hist_feature[feature_idx].x >= down_sampling_images[cur_scale_index].cols - pad_margins[cur_scale_index]) 
				||(lbp_hist_feature[feature_idx].y < pad_margins[cur_scale_index]) || (lbp_hist_feature[feature_idx].y >= down_sampling_images[cur_scale_index].rows - pad_margins[cur_scale_index]))
			{
				continue;
			}*/
			if(used_feature_cnt_map[feature_idx] == -1)
				continue;

			//FeatureAtPoint &feature_tmp = feature_collection_at_scale[used_feature_cnt++];
			FeatureAtPoint &feature_tmp = feature_collection_at_scale[used_feature_cnt_map[feature_idx]];
			feature_tmp.x = lbp_hist_feature[feature_idx].x - pad_margins[cur_scale_index];
			feature_tmp.y = lbp_hist_feature[feature_idx].y - pad_margins[cur_scale_index];

#if 0
			//Pre allocate memory
			int feature_dim = lbp_hist_feature[feature_idx].featureArray.size();
			for(int channel_idx = 0; channel_idx < symmetry_integral_images.size(); channel_idx++)
			{
				feature_dim += symmetry_feature_collections[channel_idx][feature_idx].featureArray.size();
			}
			feature_tmp.featureArray.resize(feature_dim);

			int featureArray_idx;
			for(featureArray_idx = 0; featureArray_idx < lbp_hist_feature[feature_idx].featureArray.size(); featureArray_idx++)
			{
				feature_tmp.featureArray[featureArray_idx] = lbp_hist_feature[feature_idx].featureArray[featureArray_idx];
			}
			for(int channel_idx = 0; channel_idx < symmetry_integral_images.size(); channel_idx++)
			{
				for(int i = 0;i < symmetry_feature_collections[channel_idx][feature_idx].featureArray.size(); i++)
				{
					feature_tmp.featureArray[featureArray_idx++] = symmetry_feature_collections[channel_idx][feature_idx].featureArray[i];
				}
			}
#endif

			feature_tmp.featureArray.insert(feature_tmp.featureArray.end(), 
				lbp_hist_feature[feature_idx].featureArray.begin(), lbp_hist_feature[feature_idx].featureArray.end());

			for(int channel_idx = 0; channel_idx < symmetry_integral_images.size(); channel_idx++)
			{
				//assert(top_middle_maps[channel_idx][idx].x - pad_margins[idx] == feature_temp.x && top_middle_maps[channel_idx][idx].y - pad_margins[idx] == feature_temp.y);
				feature_tmp.featureArray.insert(feature_tmp.featureArray.end(), symmetry_feature_collections[channel_idx][feature_idx].featureArray.begin(), symmetry_feature_collections[channel_idx][feature_idx].featureArray.end());
			}
		}
		//feature_collection_at_scale.erase(feature_collection_at_scale.begin() + used_feature_cnt, feature_collection_at_scale.end());
#else
		FeatureCollection feature_collection_at_scale;
		feature_collection_at_scale.resize(symmetry_feature_collections[0].size());
		int used_feature_cnt = 0;

		for(int feature_idx = 0; feature_idx < symmetry_feature_collections[0].size(); feature_idx++)
		{
			//exclude pad element
			if((symmetry_feature_collections[0][feature_idx].x < pad_margins[cur_scale_index]) || (symmetry_feature_collections[0][feature_idx].x >= down_sampling_images[cur_scale_index].cols - pad_margins[cur_scale_index]) 
				||(symmetry_feature_collections[0][feature_idx].y < pad_margins[cur_scale_index]) || (symmetry_feature_collections[0][feature_idx].y >= down_sampling_images[cur_scale_index].rows - pad_margins[cur_scale_index]))
			{
				continue;
			}

			FeatureAtPoint &feature_tmp = feature_collection_at_scale[used_feature_cnt++];
			feature_tmp.x = symmetry_feature_collections[0][feature_idx].x - pad_margins[cur_scale_index];
			feature_tmp.y = symmetry_feature_collections[0][feature_idx].y - pad_margins[cur_scale_index];
			for(int channel_idx = 0; channel_idx < symmetry_integral_images.size(); channel_idx++)
			{
				//assert(top_middle_maps[channel_idx][idx].x - pad_margins[idx] == feature_temp.x && top_middle_maps[channel_idx][idx].y - pad_margins[idx] == feature_temp.y);
				feature_tmp.featureArray.insert(feature_tmp.featureArray.end(), symmetry_feature_collections[channel_idx][feature_idx].featureArray.begin(), symmetry_feature_collections[channel_idx][feature_idx].featureArray.end());
			}
		}
		feature_collection_at_scale.erase(feature_collection_at_scale.begin() + used_feature_cnt, feature_collection_at_scale.end());
#endif
		feature_collections[idx] = feature_collection_at_scale;
	}
}

void TextlineDetector::ConstructGradientMap(const Mat &image, Mat &dst_map)
{
	// Convert color image to gray image
	Mat gray_image = image;
	GaussianBlur(gray_image, gray_image, Size(3,3), 0, 0);
	cvtColor(gray_image, gray_image, CV_RGB2GRAY);

	// Calculate grad images
	Mat grad_x, grad_y;
	Sobel(gray_image, grad_x,CV_16S, 1, 0);
	Sobel(gray_image, grad_y,CV_16S, 0, 1);
	convertScaleAbs(grad_x, grad_x);
	convertScaleAbs(grad_y, grad_y);

	// Calculate gradient magnitude
	Mat grad_magnitude;
	addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad_magnitude);
	grad_magnitude = max(min(grad_magnitude, 255), 0);

	// Discretize to bins;
	int bin = parameter.gradient_bin;
	dst_map = min(bin - 1, (bin - 1) * (grad_magnitude / 255)) + 1;
}

void TextlineDetector::ConstructLabMap(const Mat &image, Mat &dst_map)
{
	// convert color space
	Mat lab_image;
	cvtColor(image, lab_image, CV_RGB2Lab);

	// Discretize to bins
	int bin = parameter.lab_bin;
	dst_map = min(bin - 1, (bin - 1) * (lab_image / 255)) + 1;

#if _DEBUG
	double minVal, maxVal;
	cv::minMaxIdx(lab_image, &minVal, &maxVal);
	cv::minMaxIdx(dst_map, &minVal, &maxVal);
#endif
}

void TextlineDetector::ConstructTextonMap(const Mat &image, Mat &dst_map)
{
	// Convert color image to gray image
	Mat gray_image;
	cvtColor(image, gray_image, CV_RGB2GRAY);

	float *image_data = new float[gray_image.rows * gray_image.cols];
	for(int x = 0; x < gray_image.cols; x++)
	{
		int x_offset = x * gray_image.rows;
		for(int y = 0; y < gray_image.rows; y++)
		{
			image_data[y + x_offset] = gray_image.at<uchar>(y, x);
		}
	}

	// Extract LBP map
	VlLbp* lbp = vl_lbp_new(VlLbpUniform, VL_TRUE);
	int dimension = vl_lbp_get_dimension(lbp);
	float *lbp_feature = new float[gray_image.rows * gray_image.cols * dimension];
	Mat lbp_map(gray_image.rows, gray_image.cols, CV_8U);

	memset(lbp_feature, 0, sizeof(float) * gray_image.rows * gray_image.cols * dimension);
	vl_lbp_process(lbp, lbp_feature, image_data, gray_image.rows, gray_image.cols, 1);

	for(int dim = 0; dim < dimension; dim++)
	{
		int dim_offset = dim * gray_image.rows * gray_image.cols;
		for(int x = 0; x < gray_image.cols; x++)
		{
			int x_offset = x * gray_image.rows;
			for(int y = 0; y < gray_image.rows; y++)
			{
				if(0 != lbp_feature[y + x_offset + dim_offset])
				{
					lbp_map.at<uchar>(y, x) = dim + 1;
				}
			}
		}
	}
	
	/*int rows_cols = gray_image.rows * gray_image.cols;
	for(int y = 0; y < gray_image.rows; y++)
	{
		for(int x = 0; x < gray_image.cols; x++)
		{
			int dim_idx = -1;
			int offset_xy = y + x * gray_image.rows;
			for(int dim = 0; dim < dimension; dim++)
			{
				if(0 != lbp_feature[offset_xy + dim * rows_cols])
				{
					dim_idx = dim;
				}
			}
			lbp_map.at<uchar>(y, x) = dim_idx + 1;
		}
	}*/
	dst_map = lbp_map;

	//release resource
	vl_lbp_delete(lbp);
	delete(lbp_feature);
	delete(image_data);
}

void TextlineDetector::ExtractLbpHist(const Mat &integral_images, const int scale,  FeatureCollection &lbp_hist_feature)
{
	int shrink = parameter.shrink;
	if(train_flag)
	{
		shrink = 1;
	}

	// Determine valid range
	int margin = 2 * scale + 1;
	int x_min = margin;
	int x_max = integral_images.cols - margin;
	int y_min = margin;
	int y_max = integral_images.rows - margin;

	// Calculate hist at each point
	float area = (2 * scale + 1) * (4 * scale + 1);
	int feature_count = 0;
	for(int y = y_min; y <= y_max; y += shrink)
	{
		for(int x = x_min; x <= x_max; x += shrink)
		{
			int x1 = x - (scale * 2) - 1, x2 = x + (scale * 2);
			int y1 = y - scale - 1, y2 = y + scale;
			FeatureAtPoint &feature_tmp = lbp_hist_feature[feature_count++];
			feature_tmp.x = x;
			feature_tmp.y = y;
			for(int code = 0; code < parameter.texton_bin; code++)
			{
				int sub_sum = GetMatPointVal<int>(integral_images, y2, x2, code) - GetMatPointVal<int>(integral_images, y2, x1, code)
					- GetMatPointVal<int>(integral_images, y1, x2, code) + GetMatPointVal<int>(integral_images, y1, x1, code);
				feature_tmp.featureArray[code] = 1.0 * sub_sum / area;
			}
		}
	}
}

void TextlineDetector::ExtractSymmetryFeature(const Mat &integral_images, const int scale, const int bin, FeatureCollection &feature_collection)
{
	int shrink = parameter.shrink;
	if(train_flag)
	{
		shrink = 1;
	}

	// Determine valid range
	int margin = 2 * scale + 1;
	int x_min = margin;
	int x_max = integral_images.cols - margin;
	int y_min = margin;
	int y_max = integral_images.rows - margin;
	int area = (4 * scale + 1) * (scale + 1);
	int feature_count = 0;

	//Prepare feature idx
	/*map<pair<int,int>, int> feature_idx_map;
	for(int y = y_min; y <= y_max; y += shrink)
	{
		for(int x = x_min; x <= x_max; x += shrink)
		{
			feature_idx_map[make_pair(x, y)] = feature_count++;
		}
	}*/

	for(int y = y_min; y <= y_max; y += shrink)
	{
		for(int x = x_min; x <= x_max; x += shrink)
		{
			int x1 = x - (scale << 1) - 1, x2 = x + (scale << 1);
			int y1 = y - (scale << 1), y2 = y - scale, y3 = y, y4 = y + scale, y5 = y + (scale << 1);

			int top_middle_value = 0;
			int bottom_middle_value = 0;
			int middle_middle_value = 0;;

			for(int code = 0; code < bin; code++)
			{
				int y1x2 = GetMatPointVal<int>(integral_images, y1, x2, code);
				int y1x1 = GetMatPointVal<int>(integral_images, y1, x1, code);
				int y2x2 = GetMatPointVal<int>(integral_images, y2, x2, code);
				int y2x1 = GetMatPointVal<int>(integral_images, y2, x1, code);
				int y3x2 = GetMatPointVal<int>(integral_images, y3, x2, code);
				int y3x1 = GetMatPointVal<int>(integral_images, y3, x1, code);
				int y4x2 = GetMatPointVal<int>(integral_images, y4, x2, code);
				int y4x1 = GetMatPointVal<int>(integral_images, y4, x1, code);
				int y5x2 = GetMatPointVal<int>(integral_images, y5, x2, code);
				int y5x1 = GetMatPointVal<int>(integral_images, y5, x1, code);

				int sub_sum_middle_top = y3x2 - y3x1 - y2x2 + y2x1;
				int sub_sum_middle_bottom = y4x2 - y4x1 - y3x2 + y3x1;
				int sub_sum_middle = (sub_sum_middle_top + sub_sum_middle_bottom);
				int sub_sum_bottom = (y5x2 - y5x1 - y4x2 + y4x1) << 1;
				int sub_sum_top = (y2x2 - y2x1 - y1x2 + y1x1) << 1;

				//int sub_sum_top = GetMatPointVal<int>(integral_images, y2, x2, code) - GetMatPointVal<int>(integral_images, y2, x1, code)
				//	- GetMatPointVal<int>(integral_images, y1, x2, code) + GetMatPointVal<int>(integral_images, y1, x1, code);

				//int sub_sum_middle_top = GetMatPointVal<int>(integral_images, y3, x2, code) - GetMatPointVal<int>(integral_images, y3, x1, code)
				//	- GetMatPointVal<int>(integral_images, y2, x2, code) + GetMatPointVal<int>(integral_images, y2, x1, code);

				//int sub_sum_middle_bottom = GetMatPointVal<int>(integral_images, y4, x2, code) - GetMatPointVal<int>(integral_images, y4, x1, code)
				//	- GetMatPointVal<int>(integral_images, y3, x2, code) + GetMatPointVal<int>(integral_images, y3, x1, code);

				//int sub_sum_bottom = GetMatPointVal<int>(integral_images, y5, x2, code) - GetMatPointVal<int>(integral_images, y5, x1, code)
				//	- GetMatPointVal<int>(integral_images, y4, x2, code) + GetMatPointVal<int>(integral_images, y4, x1, code);

				middle_middle_value += abs(sub_sum_middle_bottom - sub_sum_middle_top);
				top_middle_value += abs(sub_sum_top - sub_sum_middle);
				bottom_middle_value += abs(sub_sum_bottom- sub_sum_middle);
				//top_middle_value +=  float((sub_sum_top - sub_sum_middle) * (sub_sum_top - sub_sum_middle)) / (2 * area * (sub_sum_top + sub_sum_middle) + EPS);
				//bottom_middle_value += float((sub_sum_bottom - sub_sum_middle) * (sub_sum_bottom - sub_sum_middle)) / (2 * area * (sub_sum_bottom + sub_sum_middle) + EPS);
				//middle_middle_value += float((sub_sum_middle_top - sub_sum_middle_bottom) * (sub_sum_middle_top - sub_sum_middle_bottom)) / (area * (sub_sum_middle_top + sub_sum_middle_bottom) + EPS);
			}
			FeatureAtPoint &feature = feature_collection[feature_count++];

			feature.x = x;
			feature.y = y;
			feature.featureArray[0] = 1.0 * top_middle_value / area;
			feature.featureArray[1] = 1.0 * bottom_middle_value / area;
			feature.featureArray[2] = 1.0 * middle_middle_value / area;
		}
	}

#if _DEBUG
#if 0
	for(int c = 0; c < bin; c++)
	{
		Mat test_mat = Mat(integral_images.rows, integral_images.cols, CV_32S);

		for(int y = y_min; y <= y_max; y++)
			for(int x = x_min; x <= x_max; x++)
			{
				test_mat.at<int>(y,x) = GetMatPointVal<int>(integral_images, y, x, c);
			}
		imshow("test_mat", test_mat);
		waitKey();
	}
#endif
#endif
}

void TextlineDetector::ExtractFeatureAtScale(const Mat &image, const int scale, FeatureCollection &features)
{
	int pad_margin = 2 * scale + 5;
	Mat image_pad;

	// Pad image
	copyMakeBorder(image, image_pad, pad_margin, pad_margin, pad_margin, pad_margin, BORDER_REPLICATE); 

	// Collect Lab color map
	Mat lab_map;
	Mat grad_map;
	Mat texton_map;

	// Construct feature maps
	ConstructLabMap(image_pad, lab_map);
	ConstructGradientMap(image_pad, grad_map);
	ConstructTextonMap(image_pad, texton_map);

#if _DEBUG
#if 0
	imshow("padimg", image_pad);
	imshow("lab_map", 255.0 * lab_map / parameter.lab_bin);
	waitKey();
#endif
#endif

	// split and merge feature maps
	vector<Mat> feature_maps;
	split(lab_map, feature_maps);
	feature_maps.push_back(grad_map.clone());
	feature_maps.push_back(texton_map.clone());

	vector<int> bins;
	bins.push_back(parameter.lab_bin);	// for l,a,b channels respectively
	bins.push_back(parameter.lab_bin);
	bins.push_back(parameter.lab_bin);
	bins.push_back(parameter.gradient_bin);
	bins.push_back(parameter.texton_bin);

	// Calculate symmerty feature.
	vector<FeatureCollection> symmetry_feature_collections;
	vector<Mat> integral_images_loop;

	// Collect feature info
	int margin = scale * 2 + 1;
	int feature_count = (image_pad.rows - 2 * margin + 1) * (image_pad.cols - 2 * margin + 1);
	for(int channel_idx = 0; channel_idx < feature_maps.size(); channel_idx++)
	{
		// Initialize feature colleciton
		FeatureCollection feature_collection;
		feature_collection.resize(feature_count);
		for(int feature_idx = 0; feature_idx < feature_count; feature_idx++)
		{
			feature_collection[feature_idx].featureArray.resize(3);	//Magic number
		}

		// Calculate integral images for each code 
		integral_images_loop.clear();

		for(int code = 1; code <= bins[channel_idx]; code++)
		{
			Mat integral_image;
			Mat feature_maps_at_code = (feature_maps[channel_idx] == code) / 255;
#if _DEBUG
#if 0
			imshow("feature_maps_at_code",255 *  feature_maps_at_code);
			waitKey();
#endif
#endif
			integral(feature_maps_at_code, integral_image);

#if _DEBUG
#if 0
			//cout << integral_image.type()<<"," << CV_32S <<endl;
			imshow("integral_image", integral_image);
			waitKey();
#endif
#endif	
			integral_images_loop.push_back(integral_image(Rect(1, 1, integral_image.cols - 1, integral_image.rows - 1)));
		}

		// merge integral images
		Mat integral_images;
		merge(integral_images_loop, integral_images);

		//extract symmetry feature
		ExtractSymmetryFeature(integral_images, scale, bins[channel_idx], feature_collection);
		symmetry_feature_collections.push_back(feature_collection);
	}

#if _DEBUG
#if 0
		ShowFeatureMap(symmetry_feature_collections, feature_maps[0].cols, feature_maps[0].rows);
#endif
#endif

	// Calculate appearance feature
	FeatureCollection lbp_hist_feature;
	lbp_hist_feature.resize(feature_count);
	for(int feature_idx = 0; feature_idx < feature_count; feature_idx++)
	{
		lbp_hist_feature[feature_idx].featureArray.resize(parameter.texton_bin);
	}

	integral_images_loop.clear();
	
#if _DEBUG
#if 0
	imshow("image", image);
	imshow("texton_map", texton_map);
	waitKey();
#endif
#endif

	for(int code = 1; code <= parameter.texton_bin; code++)
	{
		Mat integral_image;
		Mat feature_maps_at_code = (texton_map == code) / 255;
		integral(feature_maps_at_code, integral_image);
		integral_images_loop.push_back(integral_image(Rect(1, 1, integral_image.cols - 1, integral_image.rows - 1)));
	}

#if 1
	// merge integral images
	Mat integral_images;
	merge(integral_images_loop, integral_images);
	ExtractLbpHist(integral_images, scale, lbp_hist_feature);

#if _DEBUG
#if 0
	for(int code = 0; code < parameter.texton_bin; code++)
	{
		Mat lbp_show_mat(integral_images.cols, integral_images.rows, CV_32F, Scalar(0));
		for(int i = 0;i < lbp_hist_feature.size();i++)
		{
			int x = lbp_hist_feature[i].x;
			int y = lbp_hist_feature[i].y;
			lbp_show_mat.at<float>(y,x) = lbp_hist_feature[i].featureArray[code];
			//cout << lbp_hist_feature[i].featureArray[code] << endl;
		}
		double minVal, maxVal;
		minMaxIdx(lbp_show_mat, &minVal, &maxVal);
		cout << "lbp_show_mat	minVal:" << minVal << "	maxVal:"<< maxVal<<endl;
		imshow("lbp_hist_show_mat", lbp_show_mat);
		waitKey();
	}
#endif
#endif

	// Merge features
	for(int idx = 0; idx < lbp_hist_feature.size(); idx++)
	{
		//exclude pad element
		if((lbp_hist_feature[idx].x < pad_margin) || (lbp_hist_feature[idx].x >= image_pad.cols - pad_margin) 
			||(lbp_hist_feature[idx].y < pad_margin) || (lbp_hist_feature[idx].y >= image_pad.rows - pad_margin))
		{
			continue;
		}

		FeatureAtPoint feature_temp;
		vector<float> &feature_vec = feature_temp.featureArray;
		feature_temp.x = lbp_hist_feature[idx].x - pad_margin;
		feature_temp.y = lbp_hist_feature[idx].y - pad_margin;
		
		assert((lbp_hist_feature[idx].x == symmetry_feature_collections[0][idx].x) && (lbp_hist_feature[idx].y == symmetry_feature_collections[0][idx].y));

		feature_vec.insert(feature_vec.end(), lbp_hist_feature[idx].featureArray.begin(), lbp_hist_feature[idx].featureArray.end());
		
		for(int channel_idx = 0; channel_idx < feature_maps.size(); channel_idx++)
		{
			feature_vec.insert(feature_vec.end(), symmetry_feature_collections[channel_idx][idx].featureArray.begin(), symmetry_feature_collections[channel_idx][idx].featureArray.end());
		}
		features.push_back(feature_temp);
	}
#else
	for(int idx = 0;idx < symmetry_feature_collections[0].size();idx++)
	{
		//exclude pad element
		if((symmetry_feature_collections[0][idx].x < pad_margin) || (symmetry_feature_collections[0][idx].x >= image_pad.cols - pad_margin) 
			||(symmetry_feature_collections[0][idx].y < pad_margin) || (symmetry_feature_collections[0][idx].y >= image_pad.rows - pad_margin))
		{
			continue;
		}

		FeatureAtPoint feature_temp;
		vector<float> &feature_vec = feature_temp.featureArray;
		feature_temp.x = symmetry_feature_collections[0][idx].x - pad_margin;
		feature_temp.y = symmetry_feature_collections[0][idx].y - pad_margin;
		for(int channel_idx = 0; channel_idx < feature_maps.size(); channel_idx++)
		{
			feature_vec.insert(feature_vec.end(), symmetry_feature_collections[channel_idx][idx].featureArray.begin(), symmetry_feature_collections[channel_idx][idx].featureArray.end());
		}
		features.push_back(feature_temp);
	}
#endif
}

void TextlineDetector::CollectTrainingFeature(const Mat &image, const vector<Rect> &gt_boxes, FeatureCollection &features)
{
	FeatureCollection feature_buff;
	FeatureCollection feature_buff2;
	deque<FeatureCollection> feature_collection_buffer;

	feature_buff2.clear();

	// collect positive features from crop word image
	for(int gt_boxes_idx = 0; gt_boxes_idx < gt_boxes.size(); gt_boxes_idx++)
	//for(int gt_boxes_idx = 0; gt_boxes_idx < 1; gt_boxes_idx++)
	{
		Rect expanded_box;
		Rect cur_box = gt_boxes[gt_boxes_idx];

		//expand box
		expanded_box.x = max(0, cur_box.x - cur_box.height);
		expanded_box.y = max(0, cur_box.y - cur_box.height);
		expanded_box.width = min(image.cols, cur_box.x + cur_box.width + cur_box.height) - expanded_box.x;
		expanded_box.height = min(image.rows, cur_box.y + int(cur_box.height * 2)) - expanded_box.y;

		//crop expand image
		Mat croped_image(image, expanded_box);

		for(int resized_scale = scale_info.min_down_sampling_scales; resized_scale <= scale_info.max_down_sampling_scales; resized_scale++)
		{
			//resize crop iamge
			Mat resized_croped_image;
			double scale_ratio = 2.0 * resized_scale / cur_box.height;
			int h = (int)(scale_ratio * croped_image.rows);
			int w = int(croped_image.cols * scale_ratio + 0.5);
			resize(croped_image, resized_croped_image, Size(w, h)); 

#ifdef _DEBUG
#if 0 
			imshow("resized_croped_image",resized_croped_image);
			waitKey();
#endif
#endif
			//prepare init distance transform map
			Mat dist_tranform_map = Mat::ones(resized_croped_image.rows, resized_croped_image.cols, CV_8U); 

			int center_x = (int)((cur_box.x - expanded_box.x) * scale_ratio);
			int center_y = (int)((cur_box.y + int(cur_box.height / 2) - expanded_box.y) * scale_ratio);
			int end_center_x = (int)((cur_box.x + cur_box.width - expanded_box.x) * scale_ratio);
			for(; center_x <= end_center_x; center_x++)
			{
				dist_tranform_map.at<uchar>(center_y, center_x) = 0;
			}

			//distance transform
			distanceTransform(dist_tranform_map, dist_tranform_map, CV_DIST_L2, 3);

#ifdef _DEBUG	
#if 0
			// show distance transform result
			Mat dist_tranform_map_show(resized_croped_image.rows, resized_croped_image.cols, CV_8U);
			for(int x = 0;x < dist_tranform_map.cols; x++)
			{
				for(int y = 0;y < dist_tranform_map.rows; y++)
				{
					if(dist_tranform_map.at<float>(y,x) < max(1.0, parameter.positive_distance_threshold * resized_scale))
					{
						dist_tranform_map_show.at<uchar>(y,x) = 0;
					}
					else 
					{
						dist_tranform_map_show.at<uchar>(y,x) = 255;
					}
				}
			}
			imshow("test",dist_tranform_map_show);
			waitKey();
#endif
#endif

			//extract feature from crop image
			feature_buff.clear();

			ExtractFeatureAtScale(resized_croped_image, resized_scale, feature_buff);

			//collect positive features
			int positive_distance_threshold = max(1.0, parameter.positive_distance_threshold * resized_scale);
			for(int feature_buff_idx = 0; feature_buff_idx < feature_buff.size(); feature_buff_idx++)
			{
				int y = feature_buff[feature_buff_idx].y;
				int x = feature_buff[feature_buff_idx].x;
				if(dist_tranform_map.at<float>(y, x) <= positive_distance_threshold)
				{
					feature_buff[feature_buff_idx].label = 1;
					feature_buff2.push_back(feature_buff[feature_buff_idx]);
				}
			}
		}
	}

	//randomly select positive features
	random_shuffle(feature_buff2.begin(), feature_buff2.end());
	if(feature_buff2.size() > parameter.max_positive_features)
	{
		features.insert(features.end(), feature_buff2.begin(), feature_buff2.begin() + parameter.max_positive_features);
	}
	else
	{
		features.insert(features.end(), feature_buff2.begin(), feature_buff2.end());
	}

	//collect negative features from whole image
	feature_collection_buffer.clear();
	ExtractFeature(image, feature_collection_buffer);
	feature_buff2.clear();
	for(int scale_idx = 0; scale_idx < scale_info.down_sampling_scales.size(); scale_idx++)
	{	
		double resized_ratio = 1.0 / (1 << scale_info.down_sampling_indexs[scale_idx]);
		int image_h = (int)(image.rows * resized_ratio);
		int image_w = (int)(image.cols * resized_ratio);
		//distance transform
		Mat dist_tranform_map = Mat::ones(image_h, image_w, CV_8U);
		for(int gt_boxes_idx = 0; gt_boxes_idx < gt_boxes.size(); gt_boxes_idx++)
		{
			//prepare init distance transform map
			Rect cur_box = gt_boxes[gt_boxes_idx];
			int center_x = (int)(cur_box.x * resized_ratio);
			int center_y = (int)((cur_box.y + cur_box.height /2) * resized_ratio);
			int end_center_x = (int)((cur_box.x + cur_box.width) * resized_ratio);
			for(; center_x <= end_center_x; center_x++)
			{
				dist_tranform_map.at<uchar>(center_y, center_x) = 0;
			}			
		}

		//distance transform
		distanceTransform(dist_tranform_map, dist_tranform_map, CV_DIST_L2, 3);

#if _DEBUG
#if 1
		// show distance transform result
		Mat dist_tranform_map_show(image.rows, image.cols, CV_8U);
		double max_dist;
		double min_dist;
		minMaxIdx(dist_tranform_map, &min_dist, &max_dist);
		for(int x = 0;x < dist_tranform_map.cols; x++)
		{
			for(int y = 0;y < dist_tranform_map.rows; y++)
			{
				if(dist_tranform_map.at<float>(y,x) >= parameter.negative_distance_threshold)
				{
					dist_tranform_map_show.at<uchar>(y,x) = 255;
				}
				else 
				{
					dist_tranform_map_show.at<uchar>(y,x) = 0;
				}
			}
		}
		imshow("test",dist_tranform_map_show);
		waitKey();
#endif
#endif
		for(int feature_idx = 0; feature_idx < feature_collection_buffer[scale_idx].size(); feature_idx++)
		{
			int y = feature_collection_buffer[scale_idx][feature_idx].y;
			int x = feature_collection_buffer[scale_idx][feature_idx].x;
			
			if(dist_tranform_map.at<uchar>(y, x) >= parameter.negative_distance_threshold)
			{			
				feature_collection_buffer[scale_idx][feature_idx].label = 0;
				feature_buff2.push_back(feature_collection_buffer[scale_idx][feature_idx]);
			}
		}
	}
	random_shuffle(feature_buff2.begin(), feature_buff2.end());
	if(feature_buff2.size() > parameter.max_negative_features)
	{
		features.insert(features.end(), feature_buff2.begin(), feature_buff2.begin() + parameter.max_negative_features);
	}
	else
	{
		features.insert(features.end(), feature_buff2.begin(), feature_buff2.end());
	}
}

void TextlineDetector::Train(const ICDAR2011DataSet &dataset)
{
	//Create feature saving Folder
	CmFile::MkDir(dataset.feature_dir + "Train/");

	//Set train flag
	train_flag = 1;

	//prepare trainning data
	FeatureCollection features;
	for(int train_set_idx = 0; train_set_idx < dataset.train_num; train_set_idx++)
	//for(int train_set_idx = 0; train_set_idx < 60; train_set_idx++)
	{
		cout << "train_set_idx:" << train_set_idx << endl;
		time_t stime, etime;
		time(&stime);
		FeatureCollection features_tmp;

		string filename = dataset.train_set[train_set_idx];
		string full_file_path = dataset.image_dir + "Train/" + filename;
		string feature_path = dataset.feature_dir + "Train/" + filename + ".data";
		if(CmFile::FileExist(feature_path))
		{
			LoadFeature(feature_path, features_tmp);
		}
		else
		{
			// load train image
			Mat image = imread(full_file_path);
			// we suppose image is CV_8UC3 type
			assert(image.type() == CV_8UC3);

			// resize image
			int h = parameter.image_normalized_height;
			double scale_ratio = 1.0 * h / image.rows;
			int w = int(image.cols * scale_ratio + 0.5);
			resize(image, image, Size(w, h)); 

			// resize boxes
			vector<Rect> gt_train_boxes = dataset.gt_train_boxes[train_set_idx];
			for(int box_idx = 0; box_idx < gt_train_boxes.size(); box_idx++)
			{
				gt_train_boxes[box_idx].x = int(gt_train_boxes[box_idx].x * scale_ratio + 0.5);
				gt_train_boxes[box_idx].y = int(gt_train_boxes[box_idx].y * scale_ratio + 0.5);
				gt_train_boxes[box_idx].width = int(gt_train_boxes[box_idx].width * scale_ratio + 0.5);
				gt_train_boxes[box_idx].height = int(gt_train_boxes[box_idx].height * scale_ratio + 0.5);
			}

			CollectTrainingFeature(image, gt_train_boxes, features_tmp);
			SaveFeature(feature_path, features_tmp);
		}
		features.insert(features.end(), features_tmp.begin(), features_tmp.end());
		time(&etime);
		cout << "TIME:" << etime - stime << endl;
		cout << "features size:" << features_tmp.size() << endl;
	}

#if 1
	float priors[] = {1,1};
	CvRTrees random_forset_model; 
	CvRTParams params = CvRTParams(25, // max depth  
		5, // min sample count  
		0, // regression accuracy: N/A here  
		false, // compute surrogate split, no missing data  
		2, // max number of categories (use sub-optimal algorithm for larger numbers)  
		priors, // the array of priors  
		false,  // calculate variable importance  
		15,       // number of variables randomly selected at node and used to find the best split(s).  
		50,  // max number of trees in the forest  
		0.01f,               // forrest accuracy  
		CV_TERMCRIT_ITER |   CV_TERMCRIT_EPS // termination cirteria  
		); 

	random_shuffle(features.begin(), features.end());
	int dim = features[0].featureArray.size();
	int feature_count = features.size();
	int validation_feature_count = 5000;
	Mat train_data(feature_count - validation_feature_count, dim, CV_32F);
	Mat train_label(feature_count - validation_feature_count, 1, CV_32F);
	for(int feature_idx = 0; feature_idx < feature_count - validation_feature_count; feature_idx++)
	{
		for(int dim_idx = 0; dim_idx < dim; dim_idx++)
		{
			train_data.at<float>(feature_idx, dim_idx) = features[feature_idx].featureArray[dim_idx];
		}
		train_label.at<float>(feature_idx) = features[feature_idx].label;
	}

	random_forset_model.train(train_data, CV_ROW_SAMPLE, train_label, Mat(), Mat(), Mat(), Mat(), params);

	// Calculate trainning error
	Mat test_data(1, dim, CV_32F);
	float predict_correct = 0, positive_cnt = 0, negative_cnt = 0;
	for(int feature_idx = feature_count - validation_feature_count; feature_idx < feature_count; feature_idx++)
	{
		for(int dim_idx = 0; dim_idx < dim; dim_idx++)
		{
			test_data.at<float>(0, dim_idx) = features[feature_idx].featureArray[dim_idx];
		}

		if(features[feature_idx].label == random_forset_model.predict(test_data))
		{
			predict_correct++;
		}

		positive_cnt += features[feature_idx].label == 1 ? 1 : 0;
		negative_cnt += features[feature_idx].label == 0 ? 1 : 0;
	}

	cout << "Training accuracy:" << predict_correct / validation_feature_count << " pos_cnt:" << positive_cnt << " neg_cnt:" << negative_cnt << endl;
	random_forset_model.save((dataset.model_dir + "symmetry.model").c_str());
	system("pause");
#endif

#if 0
	//random_shuffle(features.begin(), features.end());
	int dim = features[0].featureArray.size();
	int feature_count = features.size();
	Mat train_data(feature_count - validation_feature_count, dim, CV_32F);
	Mat train_label(feature_count - validation_feature_count, 1, CV_32F);
	for(int feature_idx = 0; feature_idx < feature_count - 5000; feature_idx++)
	{
		for(int dim_idx = 0; dim_idx < dim; dim_idx++)
		{
			train_data.at<float>(feature_idx, dim_idx) = features[feature_idx].featureArray[dim_idx];
		}
		train_label.at<float>(feature_idx) = features[feature_idx].label;
	}

	CvSVM SVM;
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	SVM.train(train_data, train_label, Mat(), Mat(), params);
	
	// Calculate trainning error
	Mat test_data(1, dim, CV_32F);
	float predict_correct = 0, positive_cnt = 0, negative_cnt = 0;
	for(int feature_idx = feature_count - 5000; feature_idx < feature_count; feature_idx++)
	{
		for(int dim_idx = 0; dim_idx < dim; dim_idx++)
		{
			test_data.at<float>(0, dim_idx) = features[feature_idx].featureArray[dim_idx];
		}

		if(features[feature_idx].label == SVM.predict(test_data))
		{
			predict_correct++;
		}

		positive_cnt += features[feature_idx].label == 1 ? 1 : 0;
		negative_cnt += features[feature_idx].label == 0 ? 1 : 0;
	}
	cout << "Training accuracy:" << predict_correct / 5000 << " pos_cnt:" << positive_cnt << " neg_cnt:" << negative_cnt << endl;
	SVM.save(((dataset.model_dir + "symmetry.model").c_str()));
#endif
}


void TextlineDetector::NmsProb(const Mat &prob_image, Mat &nms_image)
{
	nms_image = Mat(prob_image.rows, prob_image.cols, CV_32F, Scalar(0));
	for(int x = 0; x < prob_image.cols; x++)
	{
		for(int y = 1; y < prob_image.rows - 1; y++)
		{
			if((prob_image.at<float>(y - 1, x) <= prob_image.at<float>(y, x)) && (prob_image.at<float>(y + 1, x) <= prob_image.at<float>(y, x)))
			{
				nms_image.at<float>(y,x) = prob_image.at<float>(y,x);
			}
		}
	}
}

void TextlineDetector::SelectSymmetryLinePoint(const Mat &nms_image, Mat &symmetry_line_image, double high_threshold, double low_threshold)
{
	assert(high_threshold > low_threshold);
	vector<pair<int, int>> high_prob_points;

	symmetry_line_image = Mat(nms_image.rows, nms_image.cols, CV_8U, Scalar(0));
	
	// Select high probability points
	for(int x = 0; x < nms_image.cols; x++)
	{
		for(int y = 0; y < nms_image.rows; y++)
		{
			if(nms_image.at<float>(y,x) >= high_threshold)
			{
				symmetry_line_image.at<char>(y, x) = 1;
				pair<int,int> point = make_pair(x,y);
				high_prob_points.push_back(point);
			}
		}
	}

	queue<pair<int,int>> bfs_queue;
	for(int i = 0;i < high_prob_points.size(); i++)
	{
		bfs_queue.push(high_prob_points[i]);
		while(!bfs_queue.empty())
		{
			pair<int,int> cur_point = bfs_queue.front();
			bfs_queue.pop();
			for(int delta_x = -1; delta_x <= 1; delta_x++)
			{
				for(int delta_y = -1;delta_y <= 1; delta_y++)
				{
					int next_point_x = cur_point.first + delta_x;
					int next_point_y = cur_point.second + delta_y;
					if(next_point_x >= 0 && next_point_x < nms_image.cols && next_point_y >= 0 && next_point_y < nms_image.rows)
					{
						if(nms_image.at<float>(next_point_y, next_point_x) > low_threshold && symmetry_line_image.at<char>(next_point_y, next_point_x) == 0)
						{
							symmetry_line_image.at<char>(next_point_y, next_point_x) = 1;
							bfs_queue.push(make_pair(next_point_x, next_point_y));
						}
					}
				}
			}
		}
	}
}

void TextlineDetector::LinkSymmetryLine(const Mat &symmetry_line_image, const int scale, vector<vector<cv::Point2d>> &point_group)
{
	vector<vector<cv::Point2d>> point_group1;
	Mat visit_map(symmetry_line_image.rows, symmetry_line_image.cols, CV_8U, Scalar(0));
	//First, Connecting nearby symmetry line point
	for(int x = 0; x < symmetry_line_image.cols; x++)
	{
		for(int y = 0; y < symmetry_line_image.rows; y++)
		{
			if(symmetry_line_image.at<char>(y, x) == 0 || visit_map.at<char>(y, x) == 1)
			{
				continue;
			}
			queue<cv::Point2d> bfs_queue;
			vector<cv::Point2d> group_tmp;
			cv::Point2d start_point(x, y);
			visit_map.at<char>(y, x) = 1;
			bfs_queue.push(start_point);
			group_tmp.push_back(start_point);
			while(!bfs_queue.empty())
			{
				cv::Point2d cur_point = bfs_queue.front();
				bfs_queue.pop();
				for(int delta_x = -2; delta_x <= 2; delta_x++)
				{
					for(int delta_y = -2; delta_y <= 2; delta_y++)
					{
						int next_x = cur_point.x + delta_x;
						int next_y = cur_point.y + delta_y;
						if(next_x >= 0 && next_x < symmetry_line_image.cols && next_y >= 0 && next_y < symmetry_line_image.rows)
						{
							if(symmetry_line_image.at<char>(next_y, next_x) != 0 && visit_map.at<char>(next_y, next_x) == 0)
							{
								cv::Point2d next_point(next_x, next_y);
								bfs_queue.push(next_point);
								group_tmp.push_back(next_point);
								visit_map.at<char>(next_y, next_x) = 1;
							}
						}
					}
				}
			}
			if(group_tmp.size() != 0)
			{
				point_group1.push_back(group_tmp);
			}
		}
	}

	//cout << "point_group1.size():"<< point_group1.size()<<endl;

	//Second Grouping
	vector<cv::Point2d> left_points, right_points;
	vector<int> visit_flag;
	vector<double> angles;
	visit_flag.resize(point_group1.size());
	for(int i = 0;i < point_group1.size(); i++)
	{
		cv::Point2d left_point, right_point;
		left_point = right_point = point_group1[i][0];
		for(int j = 1;j < point_group1[i].size(); j++)
		{
			if((left_point.x > point_group1[i][j].x) || (left_point.x == point_group1[i][j].x && left_point.y > point_group1[i][j].y))
			{
				left_point = point_group1[i][j];
			}

			if((right_point.x < point_group1[i][j].x) || (right_point.x == point_group1[i][j].x && right_point.y < point_group1[i][j].y))
			{
				right_point = point_group1[i][j];
			}
		}
		
		left_points.push_back(left_point);
		right_points.push_back(right_point);
		double angle = atan2(right_point.y - left_point.y, right_point.x - left_point.x);
		if(angle > CV_PI /2)
		{
			angle -= CV_PI;
		}
		if(angle < -CV_PI/2)
		{
			angle += CV_PI;
		}
		angles.push_back(angle);
	}

#if 0
	imshow("symmetry_line_image", 255 *symmetry_line_image);
	Mat show_im(symmetry_line_image.rows, symmetry_line_image.cols, CV_8UC3, Scalar(0));
	CvRNG rng;
    rng = cvRNG(cvGetTickCount());
	for(int i = 0;i < point_group1.size(); i++)
	{
		Scalar color(cvRandInt(&rng) % 256, cvRandInt(&rng) % 256, cvRandInt(&rng) % 256);
		for(int j = 0;j < point_group1[i].size();j++)
		{
			for(int k = 0;k < 3;k++)
			{
				show_im.at<Vec3b>(point_group1[i][j].y, point_group1[i][j].x)[k] = color(k);
			}
		}
	}
	imshow("show_im", show_im);
	waitKey();
#endif

	for(int i = 0;i < point_group1.size(); i++)
	{
		if(visit_flag[i] == 0)
		{
			visit_flag[i] = 1;
			queue<int> bfs_queue;
			bfs_queue.push(i);
			vector<cv::Point2d> point_group_tmp;
			point_group_tmp = point_group1[i];
			while(!bfs_queue.empty())
			{
				int cur_idx = bfs_queue.front();
				bfs_queue.pop();
				for(int next_idx = 0; next_idx < point_group1.size(); next_idx++)
				{
					double mid_x1 = 0.5 * (left_points[cur_idx].x + right_points[cur_idx].x);
					double mid_x2 = 0.5 * (left_points[next_idx].x + right_points[next_idx].x);
					double mid_y1 = 0.5 * (left_points[cur_idx].y + right_points[cur_idx].y);
					double mid_y2 = 0.5 * (left_points[next_idx].y + right_points[next_idx].y);
					double delta_x = min(abs(left_points[cur_idx].x - right_points[next_idx].x), abs(left_points[next_idx].x - right_points[cur_idx].x));
					double delta_y = abs(mid_y1 - mid_y2);
					double dist = sqrt(delta_x * delta_x + delta_y * delta_y);
					double link_angle = atan2(mid_y1 - mid_y2, mid_x1 - mid_x2);
					if(link_angle > CV_PI/2)
					{
						link_angle -= CV_PI;
					}
					if(link_angle < -CV_PI/2)
					{
						link_angle += CV_PI;
					}
					double angle_diff = max(abs(link_angle - angles[cur_idx]), abs(link_angle - angles[next_idx]));
					if(dist < scale && angle_diff < parameter.link_angle_diff_threshold && visit_flag[next_idx] == 0)
					{
						visit_flag[next_idx] = 1;
						point_group_tmp.insert(point_group_tmp.end(), point_group1[next_idx].begin(),  point_group1[next_idx].end());
						bfs_queue.push(next_idx);
					}
				}
			}
			point_group.push_back(point_group_tmp);
		}
	}

	
#if 0
	imshow("symmetry_line_image", 255 *symmetry_line_image);
	Mat show_im(symmetry_line_image.rows, symmetry_line_image.cols, CV_8UC3, Scalar(0));
	CvRNG rng;
    rng = cvRNG(cvGetTickCount());
	for(int i = 0;i < point_group.size(); i++)
	{
		Scalar color(cvRandInt(&rng) % 256, cvRandInt(&rng) % 256, cvRandInt(&rng) % 256);
		for(int j = 0;j < point_group[i].size();j++)
		{
			for(int k = 0;k < 3;k++)
			{
				show_im.at<Vec3b>(point_group[i][j].y, point_group[i][j].x)[k] = color(k);
			}
		}
	}
	imshow("show_im", show_im);
	waitKey();
#endif
}

void TextlineDetector::ResotreRect(const vector<vector<cv::Point2d>> &group_points, int w, int h, double scale_ratio, double window_scale, vector<cv::Rect> &proposals)
{
#if 0
	Mat show_im(h, w, CV_8UC3, Scalar(0));
		CvRNG rng;
    rng = cvRNG(cvGetTickCount());

	for(int i = 0;i < group_points.size();i++)
	{
		double min_x, min_y, max_x, max_y;
		min_x = max_x = group_points[i][0].x;
		min_y = max_y = group_points[i][0].y;
		Scalar color(cvRandInt(&rng) % 256, cvRandInt(&rng) % 256, cvRandInt(&rng) % 256);	
		for(int j = 0;j < group_points[i].size(); j++)
		{
			min_x = min(min_x, group_points[i][j].x);
			max_x = max(max_x, group_points[i][j].x);
			min_y = min(min_y, group_points[i][j].y);
			max_y = max(max_y, group_points[i][j].y);
			int x = min(w - 1, max(0, int(group_points[i][j].x / scale_ratio)));
			int y = min(h - 1, max(0, int(group_points[i][j].y / scale_ratio)));
			for(int k = 0;k < 3;k++)
			show_im.at<Vec3b>(y, x)[k] = color(k);
		}
		min_x = min(w - 1, max(0, int(min_x / scale_ratio)));
		min_y = min(h - 1, max(0, int(min_y / scale_ratio)));
		max_x = min(w - 1, max(0, int(max_x / scale_ratio)));
		max_y = min(h - 1, max(0, int(max_y / scale_ratio)));
		//printf("min_x:%d min_y:%d max_x:%d max_y:%d\n", (int)min_x, (int)min_y, (int)max_x, (int)max_y);
	}
	imshow("show_im", show_im);
	waitKey();

#endif
	for(int i = 0;i < group_points.size(); i++)
	{
		double min_x, min_y, max_x, max_y;
		min_x = max_x = group_points[i][0].x;
		min_y = max_y = group_points[i][0].y;
		for(int j = 1;j < group_points[i].size(); j++)
		{
			min_x = min(min_x, group_points[i][j].x);
			min_y = min(min_y, group_points[i][j].y);
			max_x = max(max_x, group_points[i][j].x);
			max_y = max(max_y, group_points[i][j].y);
		}
		min_x = min(w - 1, max(0, int(0.5 + min_x / scale_ratio)));
		min_y = min(h - 1, max(0, int(0.5 + min_y / scale_ratio - window_scale)));
		max_x = min(w - 1, max(0, int(0.5 + max_x / scale_ratio)));
		max_y = min(h - 1, max(0, int(0.5 + max_y / scale_ratio + window_scale)));
		cv::Rect rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
		proposals.push_back(rect);
	}
}

void TextlineDetector::Test(const ICDAR2011DataSet &dataset)
{
	//Set train flag
	train_flag = 0;

	CvRTrees random_forest_model;
	random_forest_model.load((dataset.model_dir + "symmetry.model").c_str());

	//CvSVM random_forset_model;
	//random_forset_model.load((dataset.model_dir + "symmetry.model").c_str());
	//processing detection
	for(int test_set_idx = 0; test_set_idx < dataset.test_num; test_set_idx++)
	{
		time_t stime, etime;
		int rf_time_sum = 0;
		time(&stime);

		string filename = dataset.test_set[test_set_idx];
		string full_file_path = dataset.image_dir + "Test/" + filename;
		string feature_path = dataset.feature_dir + "Test/" + filename + ".data";


		// load train image
		Mat image = imread(full_file_path);
		Mat resized_image;
		// we suppose image is CV_8UC3 type
		assert(image.type() == CV_8UC3);

		// resize image
		int h = parameter.image_normalized_height;
		double scale_ratio = 1.0 * h / image.rows;
		int w = int(image.cols * scale_ratio + 0.5);
		resize(image, resized_image, Size(w, h)); 

		// extract features
		deque<FeatureCollection> features_deque;
		ExtractFeature(resized_image, features_deque);
#if 0
		for(int i = 0;i < features_deque.size(); i++)
		{
			stringstream ss;
			ss << i;
			string tmp_str;
			ss >> tmp_str;
			FILE *fid;
			fid = fopen(("F:/zhengzhang/text detection/dataset/ICDAR2011/ftr" + tmp_str + ".data").c_str(), "w");
			for(int j = 0;j < features_deque[i].size(); j++)
			{
				fprintf(fid, "%d %d", features_deque[i][j].x, features_deque[i][j].y);
				for(int k = 0;k < features_deque[i][j].featureArray.size();k++)
					fprintf(fid, " %f", features_deque[i][j].featureArray[k]);
				fprintf(fid, "\n");
			}
			fclose(fid);
		}
		system("pause");
#endif
#if 0
		for(int i = 0;i < 18; i++)
		{
			stringstream ss;
			ss << i;
			string tmp_str;
			ss >> tmp_str;
			FeatureCollection fc;
			if(i != 1)
			{
				features_deque.push_back(fc);
				continue;
			}
			fc.resize(1180);
			//fc.resize(10);	
			FILE *fid;
			fid = fopen(("F:/zhengzhang/text detection/dataset/ICDAR2011/ftr" + tmp_str + ".data").c_str(), "r");
			int cnt = 0;
			while(!feof(fid))
			{
				cout << cnt<<endl;
				FeatureAtPoint fap;
				fscanf(fid,"%d %d", &fap.x, &fap.y);
				for(int j = 0;j < 73;j++)
				{
					float value;
					fscanf(fid,"%f", &value);
					fap.featureArray.push_back(value);
				}
				fc[cnt++] = fap;
				if(cnt >= 1180)
					break;
			}
			fc.erase(fc.begin() + cnt, fc.end());
			features_deque.push_back(fc);
			fclose(fid);
		}
#endif
		
		vector<cv::Rect> proposals_set;
		int cur_rows = resized_image.rows, cur_cols = resized_image.cols, cur_scale_idx = 0;
		for(int scale_idx = 0; scale_idx < scale_info.down_sampling_scales.size(); scale_idx++)
		{
			FeatureCollection &feature_collection = features_deque[scale_idx];
			int feature_count = feature_collection.size();
			int dim = feature_collection[0].featureArray.size();

			while(cur_scale_idx != scale_info.down_sampling_indexs[scale_idx])
			{
				cur_rows = (cur_rows >> 1);
				cur_cols = (cur_cols >> 1);
				cur_scale_idx++;
			}

			// construct probability map
			Mat prob_map_shrink(ceil(1.0 * cur_rows / parameter.shrink), ceil(1.0 * cur_cols / parameter.shrink), CV_32F);
			Mat prob_map(cur_rows, cur_cols, CV_32F);
			time_t rf_stime, rf_etime;
			time(&rf_stime);
			
			#pragma omp parallel for
			for(int feature_idx = 0; feature_idx < feature_count; feature_idx++)
			{
				Mat test_data(1, dim, CV_32F);
				for(int dim_idx = 0; dim_idx < dim; dim_idx++)
				{
					test_data.at<float>(0, dim_idx) = feature_collection[feature_idx].featureArray[dim_idx];
				}
				prob_map_shrink.at<float>(feature_collection[feature_idx].y / parameter.shrink, feature_collection[feature_idx].x / parameter.shrink) = random_forest_model.predict_prob(test_data);
			}			
			resize(prob_map_shrink, prob_map, Size(cur_cols, cur_rows));
			time(&rf_etime);
			rf_time_sum += rf_etime - rf_stime;

			// smooth probability map
			int kernelSize = scale_info.down_sampling_scales[scale_idx];
			kernelSize = kernelSize + 1 - (kernelSize & 1);
			cv::Size GuassianBlurKernelSize(kernelSize, kernelSize);
			GaussianBlur(prob_map, prob_map, GuassianBlurKernelSize, kernelSize/2, kernelSize/2);

			// Run nms to obtain symmetry point
			Mat nms_line_map, symmetry_line_map;
			NmsProb(prob_map, nms_line_map);
			SelectSymmetryLinePoint(nms_line_map, symmetry_line_map, 0.3, 0.1);

			vector<vector<cv::Point2d>> group_points;
			vector<cv::Rect> proposals;
			LinkSymmetryLine(symmetry_line_map, scale_info.down_sampling_scales[scale_idx], group_points);
			ResotreRect(group_points, resized_image.cols, resized_image.rows, 1.0 / (1 << cur_scale_idx), scale_info.real_scales[scale_idx], proposals);

			for(int i = 0;i < proposals.size(); i++)
			{
				proposals[i].x = min(image.cols, int(proposals[i].x / scale_ratio));
				proposals[i].y = min(image.rows, int(proposals[i].y / scale_ratio));
				proposals[i].width = min(image.cols - proposals[i].x + 1, int(proposals[i].width / scale_ratio));
				proposals[i].height = min(image.rows - proposals[i].y + 1, int(proposals[i].height / scale_ratio));
			}
			proposals_set.insert(proposals_set.end(), proposals.begin(), proposals.end());
#if 1
			Mat show_image = image.clone();
			for(int i = 0;i < proposals.size(); i++)
			{
				cv::rectangle(show_image, proposals[i], Scalar(0, 255, 255), 2,8);
			}
			
			std::stringstream ss;
			ss << scale_idx;
			string scale_idx_st;
			ss >>scale_idx_st;
			imwrite(dataset.result_dir + filename + "_" + scale_idx_st + "_bbox.jpg", show_image);
			imwrite(dataset.result_dir + filename + "_" + scale_idx_st + "_prob_map.jpg", 255 * prob_map);
#endif
		}

		FILE *fid = fopen((dataset.result_dir + filename + "_proposals.txt").c_str(), "w");
		for(int i = 0; i < proposals_set.size(); i++)
		{
			fprintf(fid, "%d %d %d %d\n", proposals_set[i].x, proposals_set[i].y, proposals_set[i].width, proposals_set[i].height);
		}
		fclose(fid);
		
		time(&etime);
		cout << "rf_time:" << rf_time_sum <<endl;
		cout << etime - stime << endl;
	}
}
 
void TextlineDetector::LoadFeature(const string &path, FeatureCollection &feature)
{
	FILE * file = fopen(path.c_str(), "rb");
	int feature_count;
	fread(&feature_count, sizeof(int), 1, file);
	feature.resize(feature_count);
	for(int feature_idx = 0; feature_idx < feature_count; feature_idx++)
	{
		int x, y, label, feature_dim;
		float feature_value;
		fread(&x, sizeof(int), 1, file);
		fread(&y, sizeof(int), 1, file);
		fread(&label, sizeof(int), 1, file);
		fread(&feature_dim, sizeof(int), 1, file);

		for(int dim_idx = 0; dim_idx < feature_dim; dim_idx++)
		{
			fread(&feature_value, sizeof(float), 1, file);
			feature[feature_idx].featureArray.push_back(feature_value);
		}
		feature[feature_idx].x = x;
		feature[feature_idx].y = y;
		feature[feature_idx].label = label;
	}
	fclose(file);
}

void TextlineDetector::SaveFeature(const string &path, FeatureCollection &feature)
{
	FILE * file = fopen(path.c_str(), "wb");
	int feature_count = feature.size();
	fwrite(&feature_count, sizeof(int), 1, file);
	for(int feature_idx = 0; feature_idx < feature_count; feature_idx++)
	{
		int feature_dim = feature[feature_idx].featureArray.size();
		fwrite(&(feature[feature_idx].x), sizeof(int), 1, file);
		fwrite(&(feature[feature_idx].y), sizeof(int), 1, file);
		fwrite(&(feature[feature_idx].label), sizeof(int), 1, file);
		fwrite(&feature_dim, sizeof(int), 1, file);
		fwrite(feature[feature_idx].featureArray.data(), sizeof(float), feature_dim, file);
	}
	fclose(file);
}

void TextlineDetector::InitialzeScaleInfo()
{
	double scaling_ratio = pow(2.0, 1.0 / parameter.scale_num_octiva);
	int scale_cnt = 0;

	scale_info.min_down_sampling_scales = 32767;
	scale_info.max_down_sampling_scales = 0;
	scale_info.key_down_sampling_scale_num = parameter.max_scale - parameter.min_scale + 1;

	for(int key_scale = parameter.min_scale; key_scale <= parameter.max_scale; key_scale++)
	{
		int base_scale = 1 << (key_scale - parameter.min_scale);
		for(int octiva_idx = 0; octiva_idx < parameter.scale_num_octiva; octiva_idx++)
		{
			double real_scale = (1 << key_scale) * pow(scaling_ratio, octiva_idx);
			scale_info.real_scales.push_back(real_scale);
			scale_info.down_sampling_indexs.push_back(key_scale - parameter.min_scale);
			scale_info.down_sampling_scales.push_back((int)(real_scale / base_scale));
			scale_info.min_down_sampling_scales = min(scale_info.min_down_sampling_scales, scale_info.down_sampling_scales[scale_cnt]);
			scale_info.max_down_sampling_scales = max(scale_info.min_down_sampling_scales, scale_info.down_sampling_scales[scale_cnt]);
			scale_cnt++;

			cout << "real_scale:" << real_scale << ",down_sampling_scale:" << scale_info.down_sampling_scales[scale_cnt-1]  << ",key_index:"<<scale_info.down_sampling_indexs[scale_cnt-1]<<endl;
		}


	}
	cout << "min_down_sampling_scales" << scale_info.min_down_sampling_scales << ", max_down_sampling_scales" << scale_info.max_down_sampling_scales << endl;
}

void TextlineDetector::ShowFeatureMap(const vector<FeatureCollection> &feature, const int width, const int height)
{
	assert(feature.size() != 0);

	vector<Mat> im_array;
	for(int i = 0;i < 3;i++)
	{
		im_array.push_back(Mat(height, width, CV_32FC3, Scalar(0)));
	}

	for(int i = 0; i < min(3,(int)feature.size()); i++)
	{
		const FeatureCollection &feature_at_channel = feature[i];
		for(int feature_idx = 0; feature_idx < feature_at_channel.size(); feature_idx++)
		{
			int x = feature_at_channel[feature_idx].x;
			int y = feature_at_channel[feature_idx].y;

			for(int idx = 0; idx < feature_at_channel[feature_idx].featureArray.size(); idx++)
			{
				im_array[idx].at<cv::Vec3f>(y, x)[i] =  feature_at_channel[feature_idx].featureArray[idx];
			}

		}
	}

	double minVal, maxVal;
	cv::minMaxIdx(im_array[0], &minVal, &maxVal);
	cout << minVal <<","<<maxVal<<endl;
	imshow("feature_map[0]", im_array[0]);
	imshow("feature_map[1]", im_array[1]);
	imshow("feature_map[2]", im_array[2]);
	waitKey();
}

TextlineDetector::TextlineDetector(const TextlineDetectorParameter &_parameter)
{
	parameter = _parameter;
	omp_set_num_threads(parameter.core_num);
	InitialzeScaleInfo();
}


TextlineDetector::TextlineDetector(void)
{
	parameter.image_normalized_height = 800;
	parameter.scale_num_octiva = 3;
	parameter.min_scale = 3;
	parameter.max_scale = 8;
	parameter.negative_distance_threshold = 15;
	parameter.positive_distance_threshold = 0.2;
	parameter.max_negative_features = 3000;
	parameter.max_positive_features = 2000;
	parameter.lab_bin = 32;
	parameter.gradient_bin = 32;
	parameter.texton_bin = 58;
	parameter.link_angle_diff_threshold = CV_PI / 4;
	parameter.core_num = 16;
	parameter.shrink = 2;
	omp_set_num_threads(parameter.core_num);
	InitialzeScaleInfo();
}


TextlineDetector::~TextlineDetector(void)
{
}
