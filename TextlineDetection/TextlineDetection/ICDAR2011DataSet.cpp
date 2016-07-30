#include "stdafx.h"
#include "CmFile.h"
#include "ICDAR2011DataSet.h"

void ICDAR2011DataSet::LoadAnnotation(const string file_path, vector<vector<Rect>> &boxes)
{
	vector<Rect> _boxes;
	ifstream in(file_path);

	if(false == in.is_open())
	{
		cout << "err in ICDAR2011DataSet::LoadAnnotation" << endl;
	}
	else
	{
		string str_buffer;
		while(getline(in, str_buffer))
		{
			int x1, y1, x2, y2;
			sscanf(str_buffer.c_str(), "%d,%d,%d,%d", &x1, &y1, &x2, &y2);
			_boxes.push_back(Rect(x1, y1, x2 - x1, y2 - y1));
		}
	}
	boxes.push_back(_boxes);
	in.close();
}

void ICDAR2011DataSet::LoadCharacterAnnotation(const string file_path, vector<vector<Rect>> &boxes)
{
	vector<Rect> _boxes;
	ifstream in(file_path);

	if(false == in.is_open())
	{
		cout << "err in ICDAR2011DataSet::LoadAnnotation" << endl;
	}
	else
	{
		string str_buffer;
		while(getline(in, str_buffer))
		{
			int x1, y1, x2, y2;
			sscanf(str_buffer.c_str(), "%d %d %d %d", &x1, &y1, &x2, &y2);
			if(x2 == x1 || y2 == y1)
			{
				continue;
			}
			_boxes.push_back(Rect(x1, y1, x2 - x1, y2 - y1));
		}
	}
	boxes.push_back(_boxes);
	in.close();
}

ICDAR2011DataSet::ICDAR2011DataSet(const string &_work_dir)
{
	work_dir = _work_dir;
	image_dir = work_dir + "Images/";
	annotation_dir = work_dir + "Annotation/";
	feature_dir = work_dir + "Feature/";
	result_dir = work_dir + "Result/";
	model_dir = work_dir + "Model/";
	CmFile::MkDir(feature_dir);
	CmFile::MkDir(result_dir);
	CmFile::MkDir(model_dir);
	
	CmFile::GetNames(image_dir + "train/", "*.jpg", train_set);
	CmFile::GetNames(image_dir + "test/", "*.jpg", test_set);

	train_num = train_set.size();
	test_num = test_set.size();

	//load annotation
	for(int i = 0; i < train_num; i++)
	{
		string name_ne = CmFile::GetNameNE(train_set[i]);
		LoadCharacterAnnotation(annotation_dir + "train/" +  name_ne + "_GT.txt", gt_train_boxes);
	}

	for(int i = 0; i < test_num; i++)
	{
		string name_ne = CmFile::GetNameNE(test_set[i]);
		LoadAnnotation(annotation_dir + "test/" + "gt_" + name_ne + ".txt", gt_test_boxes);
	}
}

ICDAR2011DataSet::ICDAR2011DataSet(void)
{
}


ICDAR2011DataSet::~ICDAR2011DataSet(void)
{
}
