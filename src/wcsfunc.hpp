#ifndef W2XC_WCSFUNC_HPP
#define W2XC_WCSFUNC_HPP

#include <cstring>
#include <string>
#include <algorithm>

std::wstring to_wcs(std::string str);
std::string to_mbs(std::wstring str);

std::string ReplaceString(std::string subject, const std::string& search, const std::string& replace);
std::wstring ReplaceString(std::wstring subject, const std::wstring& search, const std::wstring& replace);

std::string basename(const std::string& str);
std::wstring basename(const std::wstring& str);

std::string get_extension(const std::string& str);
std::wstring get_extension(const std::wstring& str);

std::string trim(const std::string& str);
std::wstring trim(const std::wstring& str);

#endif
