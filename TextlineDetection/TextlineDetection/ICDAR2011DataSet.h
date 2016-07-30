#pragma once

class ICDAR2011DataSet
{
public:
	string work_dir;
	string image_dir;
	string annotation_dir;
	string feature_dir;
	string result_dir;
	string model_dir;

	int train_num, test_num;
	vector<string> train_set, test_set;
	vector<vector<Rect>> gt_train_boxes, gt_test_boxes;

private:
	void LoadAnnotation(const string file_path, vector<vector<Rect>> &boxes);
	void LoadCharacterAnnotation(const string file_path, vector<vector<Rect>> &boxes);

public:
	ICDAR2011DataSet(void);
	ICDAR2011DataSet(const string &work_dir);
	~ICDAR2011DataSet(void);
};

