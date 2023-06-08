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

#ifndef __TCHAR_H__
#define __TCHAR_H__

#if defined(_WIN32) && defined(_UNICODE)
	#define	_T(x)	L ## x
#else
	#define	_T(x)	x
#endif


#if defined(_WIN32) && defined(_UNICODE)
	#define	_tmain		wmain
	#define	_tcslen		wcslen
	#define	_tcscat		wcscat
	#define _tcschr		wcschr
	#define	_tcscpy		wcscpy
	#define	_tcsncpy	wcsncpy
	#define	_tcscmp		wcscmp
	#define	_tcsncmp	wcsncmp
	#define _tcsstr		wcsstr
	#define _tcsrev		_wcsrev
	#define	_tprintf	wprintf
	#define	_stprintf	swprintf
	#define	_tscanf		wscanf
	#define _tfopen		_wfopen
	#define	_fgetts		fgetws
	#define	_fputts		fputws
	#define _totlower	towlower
	#define _to_tstring	to_wstring
#else
	#define	_tmain		main
	#define	_tcslen		strlen
	#define	_tcscat		strcat
	#define _tcschr		strchr
	#define	_tcscpy		strcpy
	#define	_tcsncpy	strncpy
	#define	_tcscmp		strcmp
	#define _tcsncmp	strncmp
	#define _tcsstr		strstr
	#define _tcsrev		strrev
	#define	_tprintf	printf
	#define	_stprintf	sprintf
	#define	_tscanf		scanf
	#define _tfopen		fopen
	#define _fgetts		fgets
	#define	_fputts		fputs
	#define _totlower	tolower
	#define _to_tstring	to_string
#endif

#if defined(_WIN32) && defined(_UNICODE)
	typedef	wchar_t	TCHAR;
#else
	typedef	char	TCHAR;
#endif

#endif
