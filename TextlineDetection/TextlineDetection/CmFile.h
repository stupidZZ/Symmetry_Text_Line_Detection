#pragma once
#include "stdafx.h"

#define _S(str) ((str).c_str())
#define CV_Assert_(expr, args) \
{\
	if(!(expr)) {\
	string msg = cv::format args; \
	printf("%s in %s:%d\n", msg.c_str(), __FILE__, __LINE__); \
	cv::error(cv::Exception(CV_StsAssert, msg, __FUNCTION__, __FILE__, __LINE__) ); }\
}

struct CmFile
{
	static string BrowseFile(const char* strFilter = "Images (*.jpg;*.png)\0*.jpg;*.png\0All (*.*)\0*.*\0\0", bool isOpen = true);
	static string BrowseFolder(); 

	static inline string GetFolder(const string& path);
	static inline string GetName(const string& path);
	static inline string GetNameNE(const string& path);
	static inline string GetPathNE(const string& path);

	// Get file names from a wildcard. Eg: GetNames("D:\\*.jpg", imgNames);
	static int GetNames(const string &nameW, vector<string> &names, string &dir = string());
	static int GetNames(const string& rootFolder, const string &fileW, vector<string> &names);
	static int GetNamesNE(const string& nameWC, vector<string> &names, string &dir = string(), string &ext = string());
	static int GetNamesNE(const string& rootFolder, const string &fileW, vector<string> &names);
	static inline string GetExtention(const string name);

	static inline bool FileExist(const string& filePath);
	static inline bool FilesExist(const string& fileW);
	static inline bool FolderExist(const string& strPath);

	static inline string GetWkDir();

	static BOOL MkDir(const string&  path);

	// Eg: RenameImages("D:/DogImages/*.jpg", "F:/Images", "dog", ".jpg");
	static int Rename(const string& srcNames, const string& dstDir, const char* nameCommon, const char* nameExt);

	static inline void RmFile(const string& fileW);
	static void RmFolder(const string& dir);
	static void CleanFolder(const string& dir, bool subFolder = false);

	static int GetSubFolders(const string& folder, vector<string>& subFolders);

	inline static BOOL Copy(const string &src, const string &dst, BOOL failIfExist = FALSE);
	inline static BOOL Move(const string &src, const string &dst, DWORD dwFlags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED | MOVEFILE_WRITE_THROUGH);
	static BOOL Move2Dir(const string &srcW, const string dstDir);
	static BOOL Copy2Dir(const string &srcW, const string dstDir);

	//Load mask image and threshold thus noisy by compression can be removed
	static Mat LoadMask(const string& fileName);

	static void WriteNullFile(const string& fileName) {FILE *f = fopen(_S(fileName), "w"); fclose(f);}

	static void ChkImgs(const string &imgW);

	static void RunProgram(const string &fileName, const string &parameters = "", bool waiteF = false, bool showW = true);

	static void SegOmpThrdNum(double ratio = 0.8);

	// Copy files and add suffix. e.g. copyAddSuffix("./*.jpg", "./Imgs/", "_Img.jpg")
	static void copyAddSuffix(const string &srcW, const string &dstDir, const string &dstSuffix);

	static vector<string> loadStrList(const string &fName);
	static bool writeStrList(const string &fName, const vector<string> &strs);
};

/************************************************************************/
/* Implementation of inline functions                                   */
/************************************************************************/
string CmFile::GetFolder(const string& path)
{
	return path.substr(0, path.find_last_of("\\/")+1);
}

string CmFile::GetName(const string& path)
{
	int start = path.find_last_of("\\/")+1;
	int end = path.find_last_not_of(' ')+1;
	return path.substr(start, end - start);
}

string CmFile::GetNameNE(const string& path)
{
	int start = path.find_last_of("\\/")+1;
	int end = path.find_last_of('.');
	if (end >= 0)
		return path.substr(start, end - start);
	else
		return path.substr(start,  path.find_last_not_of(' ')+1 - start);
}

string CmFile::GetPathNE(const string& path)
{
	int end = path.find_last_of('.');
	if (end >= 0)
		return path.substr(0, end);
	else
		return path.substr(0,  path.find_last_not_of(' ') + 1);
}

string CmFile::GetExtention(const string name)
{
	return name.substr(name.find_last_of('.'));
}

BOOL CmFile::Copy(const string &src, const string &dst, BOOL failIfExist)
{
	return ::CopyFileA(src.c_str(), dst.c_str(), failIfExist);
}

BOOL CmFile::Move(const string &src, const string &dst, DWORD dwFlags)
{
	return MoveFileExA(src.c_str(), dst.c_str(), dwFlags);
}

void CmFile::RmFile(const string& fileW)
{ 
	vector<string> names;
	string dir;
	int fNum = CmFile::GetNames(fileW, names, dir);
	for (int i = 0; i < fNum; i++)
		::DeleteFileA(_S(dir + names[i]));
}


// Test whether a file exist
bool CmFile::FileExist(const string& filePath)
{
	if (filePath.size() == 0)
		return false;

	return  GetFileAttributesA(_S(filePath)) != INVALID_FILE_ATTRIBUTES; // ||  GetLastError() != ERROR_FILE_NOT_FOUND;
}

bool CmFile::FilesExist(const string& fileW)
{
	vector<string> names;
	int fNum = GetNames(fileW, names);
	return fNum > 0;
}

string CmFile::GetWkDir()
{	
	string wd;
	wd.resize(1024);
	DWORD len = GetCurrentDirectoryA(1024, &wd[0]);
	wd.resize(len);
	return wd;
}

bool CmFile::FolderExist(const string& strPath)
{
	int i = (int)strPath.size() - 1;
	for (; i >= 0 && (strPath[i] == '\\' || strPath[i] == '/'); i--)
		;
	string str = strPath.substr(0, i+1);

	WIN32_FIND_DATAA  wfd;
	HANDLE hFind = FindFirstFileA(_S(str), &wfd);
	bool rValue = (hFind != INVALID_HANDLE_VALUE) && (wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);   
	FindClose(hFind);
	return rValue;
}

/************************************************************************/
/*                   Implementations                                    */
/************************************************************************/