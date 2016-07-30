#pragma once
#include "stdafx.h"

#define EPS 1e-6

struct FeatureAtPoint
{
	int label;
	int x, y;
	vector<float> featureArray;
};

typedef vector<FeatureAtPoint> FeatureCollection;

template<typename ClassName>
ClassName GetMatPointVal(const Mat &src, int row, int col, int channel)
{
	const ClassName* cur_row = src.ptr<ClassName>(row);
	return *(cur_row + col * src.channels() + channel);
}

template<typename ClassName>
ClassName *GetMatPointPtr(Mat &src, int row, int col, int channel)
{
	ClassName* cur_row = src.ptr<ClassName>(row);
	return cur_row + col * src.channels() + channel;
}