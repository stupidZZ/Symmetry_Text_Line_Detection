// TextlineDetection.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ICDAR2011DataSet.h";
#include "TextlineDetector.h";

int _tmain(int argc, _TCHAR* argv[])
{
	if(argc != 2)
	{
		cout << "Missing config path" << endl;
		return 0;
	}
	int runtime_path_len = WideCharToMultiByte(CP_ACP, 0, argv[0], -1, NULL, 0, NULL, NULL);
	int config_name_len = WideCharToMultiByte(CP_ACP, 0, argv[1], -1, NULL, 0, NULL, NULL);
	char *ptr_runtime_path = new char[runtime_path_len +1];
	char *ptr_config_name = new char[config_name_len + 1];
	WideCharToMultiByte(CP_ACP, 0, argv[0], -1, ptr_runtime_path, runtime_path_len, NULL, NULL);
	WideCharToMultiByte(CP_ACP, 0, argv[1], -1, ptr_config_name, config_name_len, NULL, NULL);

	string runtime_path = CmFile::GetFolder(ptr_runtime_path);
	string config_name(ptr_config_name);
	
	char string_buffer[1024];
	string dataset_path;
	int mode;
	TextlineDetectorParameter parameter;
	FILE *fid = fopen((runtime_path + config_name).c_str(), "r");
	if(fid == NULL)
	{
		cout << "Open config file failed!"<<endl;
		return 0;
	}
	fgets(string_buffer, 1024, fid); 
	dataset_path = CmFile::GetFolder(string_buffer);
	fscanf(fid, "%d %d %d %d %d %d %lf %d %d %d %d %d %lf %d %d",	
			&mode,
			&parameter.image_normalized_height,
			&parameter.scale_num_octiva,
			&parameter.min_scale,
			&parameter.max_scale,
			&parameter.negative_distance_threshold,
			&parameter.positive_distance_threshold,
			&parameter.max_negative_features,
			&parameter.max_positive_features,
			&parameter.lab_bin,
			&parameter.gradient_bin,
			&parameter.texton_bin,
			&parameter.link_angle_diff_threshold,
			&parameter.core_num,
			&parameter.shrink);
	fclose(fid);

	ICDAR2011DataSet dataset(dataset_path);
	TextlineDetector detector(parameter);
	if(mode == 1)
	{
		detector.Train(dataset);
	}
	else
	{
		detector.Test(dataset);
	}
	return 0;
}

