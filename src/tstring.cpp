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


#include "tstring.hpp"

std::string wstr2str(std::wstring ws){
	std::setlocale(LC_ALL, "en_US.utf8");
	char *buf = new char[ws.size()];
	size_t num_chars = wcstombs(buf, ws.c_str(), ws.size());
	
	std::string s( buf, num_chars );
	delete[] buf;
	return s;
}

std::wstring str2wstr(std::string s){
	std::setlocale(LC_ALL, "en_US.utf8");
	wchar_t *buf = new wchar_t[s.size()];
	size_t num_chars = mbstowcs(buf, s.c_str(), s.size());
	
	std::wstring ws( buf, num_chars );
	delete[] buf;
	return ws;
}