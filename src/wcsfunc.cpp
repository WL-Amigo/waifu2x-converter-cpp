/*
* wcsfunc.cpp
*
*  Created on: 2018/12/27
*	  Author: YukihoAA
*
*/

#include "wcsfunc.hpp"


std::wstring to_wcs(std::string str){
	std::wstring result;
	result.assign(str.begin(), str.end());
	return result;
}

std::string to_mbs(std::wstring str){
	std::string result;
	result.assign(str.begin(), str.end());
	return result;
}


std::string ReplaceString(std::string subject, const std::string& search, const std::string& replace) {
	size_t pos = 0;
	while ((pos = subject.find(search, pos)) != std::string::npos) {
		subject.replace(pos, search.length(), replace);
		pos += replace.length();
	}
	return subject;
}

std::wstring ReplaceString(std::wstring subject, const std::wstring& search, const std::wstring& replace) {
	size_t pos = 0;
	while ((pos = subject.find(search, pos)) != std::wstring::npos) {
		subject.replace(pos, search.length(), replace);
		pos += replace.length();
	}
	return subject;
}


std::string basename(const std::string& str) {
	size_t found = str.find_last_of("/\\");
	return str.substr(found + 1);
}

std::wstring basename(const std::wstring& str) {
	size_t found = str.find_last_of(L"/\\");
	return str.substr(found + 1);
}


std::string get_extension(const std::string& str) {
	size_t found = str.find_last_of('.');
	if(found == std::string::npos)
		return std::string("");
	return str.substr(found + 1);
}

std::wstring get_extension(const std::wstring& str) {
	size_t found = str.find_last_of(L'.');
	if(found == std::wstring::npos)
		return std::wstring(L"");
	return str.substr(found + 1);
}


std::string trim(const std::string& str)
{
	size_t first = str.find_first_not_of(' ');
	if (std::string::npos == first)
	{
		return str;
	}
	size_t last = str.find_last_not_of(' ');
	return str.substr(first, (last - first + 1));
}

std::wstring trim(const std::wstring& str)
{
	size_t first = str.find_first_not_of(L' ');
	if (std::wstring::npos == first)
	{
		return str;
	}
	size_t last = str.find_last_not_of(L' ');
	return str.substr(first, (last - first + 1));
}
