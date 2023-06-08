/*
* The MIT License (MIT)
* This file is part of waifu2x-converter-cpp
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#ifndef __TSTRING_H__
#define __TSTRING_H__

#include <string>
#include <clocale>
#include "tchar.h"

std::string wstr2str(std::wstring ws);
std::wstring str2wstr(std::string s);

#if defined(_WIN32) && defined(_UNICODE)
	typedef	std::wstring		_tstring;
	typedef	std::wstringstream	_tstringstream;
	#define _tstr2wstr(X) X;
	#define _tstr2str(X) wstr2str(X);
	#define _wstr2tstr(X) X;
	#define _str2tstr(X) str2wstr(X);
#else
	typedef	std::string			_tstring;
	typedef	std::stringstream	_tstringstream;
	#define _tstr2wstr(X) str2wstr(X);
	#define _tstr2str(X) X;
	#define _wstr2tstr(X) wstr2str(X);
	#define _str2tstr(X) X;
#endif

#if defined(_WIN32) && defined(_UNICODE)
	#define	TSTRING_METHOD	wstring
#else
	#define	TSTRING_METHOD	string
#endif

#endif
