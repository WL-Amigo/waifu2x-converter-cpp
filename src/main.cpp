/*
* main.cpp
*   (ここにファイルの簡易説明を記入)
*
*  Created on: 2015/05/24
*      Author: wlamigo
*
*   (ここにファイルの説明を記入)
*/

#include <cassert>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <string>
#include <cmath>
#include <deque>
#include <map>
#include <stdio.h>
#include "tclap/CmdLine.h"
#include "sec.hpp"
#include <experimental/filesystem>

#include "w2xconv.h"

#ifndef DEFAULT_MODELS_DIRECTORY
#define DEFAULT_MODELS_DIRECTORY "models_rgb"
#endif

#define _VERSION "5.2"

#ifdef HAVE_OPENCV
#include <opencv2/opencv.hpp>
#endif

class CustomFailOutput : public TCLAP::StdOutput {
public:
    virtual void failure( TCLAP::CmdLineInterface& _cmd, TCLAP::ArgException& e ) override {
		std::string progName = _cmd.getProgramName();

		std::cerr << "PARSE ERROR: " << e.argId() << std::endl
				  << "             " << e.error() << std::endl << std::endl;

		if ( _cmd.hasHelpAndVersion() )
			{
				std::cerr << "Brief USAGE: " << std::endl;

				_shortUsage( _cmd, std::cerr );	

				std::cerr << std::endl << "For complete USAGE and HELP type: " 
						  << std::endl << "   " << progName << " --help" 
						  << std::endl << std::endl;
				std::cerr << "Waifu2x OpenCV - Version " << _VERSION << " - https://github.com/DeadSix27/waifu2x-converter-cpp" << std::endl << std::endl;
				std::cerr << "If you find issues or need help, visit: https://github.com/DeadSix27/waifu2x-converter-cpp/issues" << std::endl << std::endl;
			}
		else
			usage(_cmd);

		throw TCLAP::ExitException(1);
    }
};


namespace fs = std::experimental::filesystem;

struct convInfo {
	std::string mode;
	int NRLevel;
	double scaleRatio;
	int blockSize;
	W2XConv* converter;
	convInfo(std::string mode, int NRLevel, double scaleRatio, int blockSize, W2XConv* converter) :mode(mode), NRLevel(NRLevel), scaleRatio(scaleRatio), blockSize(blockSize), converter(converter) {};
};

static void
dump_procs()
{
	const W2XConvProcessor *procs;
	int num_proc;
	procs = w2xconv_get_processor_list(&num_proc);

	for (int i = 0; i < num_proc; i++) {
		const W2XConvProcessor *p = &procs[i];
		const char *type;
		switch (p->type) {
		case W2XCONV_PROC_HOST:
			switch (p->sub_type) {
			case W2XCONV_PROC_HOST_AVX:
				type = "AVX";
				break;
			case W2XCONV_PROC_HOST_FMA:
				type = "FMA";
				break;
			case W2XCONV_PROC_HOST_SSE3:
				type = "SSE3";
				break;
			case W2XCONV_PROC_HOST_NEON:
				type = "NEON";
				break;
			default:
				type = "OpenCV";
				break;
			}
			break;

		case W2XCONV_PROC_CUDA:
			type = "CUDA";
			break;

		case W2XCONV_PROC_OPENCL:
			type = "OpenCL";
			break;

		default:
			type = "??";
			break;
		}

		printf("%4d: %-45s(%-10s): num_core=%d\n",
			i,
			p->dev_name,
			type,
			p->num_core);
	}
}

void check_for_errors(W2XConv* converter, int error) {
	if (error) {
		char *err = w2xconv_strerror(&converter->last_error);
		std::string errorMessage(err);
		w2xconv_free(err);
		throw std::runtime_error(errorMessage);
	}
}


std::string ReplaceString(std::string subject, const std::string& search, const std::string& replace) {
	size_t pos = 0;
	while ((pos = subject.find(search, pos)) != std::string::npos) {
		subject.replace(pos, search.length(), replace);
		pos += replace.length();
	}
	return subject;
}
std::string basename(const std::string& str) {
	size_t found = str.find_last_of("/\\");
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


std::map<std::string,bool> opencv_formats = {
	{"BMP",  false},
	{"DIB",  false},
	{"PBM",  false},
	{"PGM",  false},
	{"PPM",  false},
	{"SR",  false},
	{"RAS",  false},
	{"PNG",  false},
	{"JPEG", false},
	{"JPG", false},
	{"JPE", false},
	{"JP2", false},
	{"WEBP", false},
	{"TIFF", false}
};

bool check_output_extension(std::string extension) {
	for(std::string::iterator it = extension.begin(); it != extension.end(); ++it){
		*it = std::toupper(*it);
	}
	auto index = opencv_formats.find(extension);
	if (index != opencv_formats.end()) {
		return index->second;
	}
	return false;
}


std::string generate_output_location(std::string inputFileName, std::string outputFileName, std::string mode, int NRLevel, double scaleRatio) {

	size_t lastSlashPos = outputFileName.find_last_of("/\\");
	size_t lastDotPos = outputFileName.find_last_of('.');

	if (strcmp(outputFileName.c_str(), "auto")==0) {
		outputFileName = inputFileName;
		size_t tailDot = outputFileName.find_last_of('.');
		if (tailDot != std::string::npos)
			outputFileName.erase(tailDot, outputFileName.length());
		outputFileName = outputFileName + "_[" + ReplaceString(mode, "noise_scale", "NS") + "-";
		//std::string &mode = mode;
		if (mode.find("noise") != mode.npos) {
			outputFileName = outputFileName + "L" + std::to_string(NRLevel) + "]";
		}
		if (mode.find("scale") != mode.npos) {
			outputFileName = outputFileName + "[x" + std::to_string(scaleRatio) + "]";
		}
		outputFileName += ".png";
	}
	else if (outputFileName.back() == '/' || outputFileName.back() == '\\') {
		//outputFileName = output folder or "auto/"
		if ((!fs::is_directory(outputFileName))) {
			fs::create_directories(outputFileName);
		}
		//We pass tmp into generate_output_location because we will use the default way of naming processed files.
		//We will remove everything, in the tmp string, prior to the last slash to get the filename.
		//This removes all contextual information about where a file originated from if "recursive_directory" was enabled.
		std::string tmp = generate_output_location(inputFileName, "auto", mode, NRLevel, scaleRatio);
		//tmp = full formatted output file path
		size_t lastSlash = tmp.find_last_of('/');
		if (lastSlash != std::string::npos){
			tmp.erase(0, lastSlash);
		}

		outputFileName += basename(tmp);
	}
	else if (lastDotPos == std::string::npos || lastSlashPos != std::string::npos && lastDotPos < lastSlashPos) {
		//e.g. ./test.d/out needs to be changed to ./test.d/out.png
		outputFileName += ".png";
	}
	else if (lastSlashPos == std::string::npos || lastDotPos > lastSlashPos) {
		//We may have a regular output file here or something went wrong.
		//outputFileName is already what it should be thus nothing needs to be done.
		#ifdef HAVE_OPENCV
		if(check_output_extension(outputFileName.substr(lastDotPos+1))==false){
			throw std::runtime_error("Unsupported output extension. outputFileName:" + outputFileName + " extension:" +outputFileName.substr(lastDotPos+1));
		}
		#endif
	}
	else {
		throw std::runtime_error("An unknown 'outputFileName' has been inserted into generate_output_location. outputFileName: " + outputFileName);
	}
	return outputFileName;
}


void convert_file(convInfo info, fs::path inputName, fs::path output) {
	//std::cout << "Operating on: " << fs::absolute(inputName).string() << std::endl;
	std::string outputName = generate_output_location(fs::absolute(inputName).string(), output.string(), info.mode, info.NRLevel, info.scaleRatio);

	int _nrLevel = 0;

	if (strcmp(info.mode.c_str(), "noise")==0 || strcmp(info.mode.c_str(), "noise_scale")==0) {
		_nrLevel = info.NRLevel;
	}

	double _scaleRatio = 1;
	if (strcmp(info.mode.c_str(), "scale")==0 || strcmp(info.mode.c_str(), "noise_scale")==0) {
		_scaleRatio = info.scaleRatio;
	}

	int error = w2xconv_convert_file(info.converter,
		outputName.c_str(),
		fs::absolute(inputName).string().c_str(),
		_nrLevel,
		_scaleRatio, info.blockSize);

	check_for_errors(info.converter, error);
}

	
//check for opencv formats
void check_opencv_formats()
{
	std::istringstream iss(cv::getBuildInformation());

	for (std::string line; std::getline(iss, line); )
	{
		std::vector<std::string> strings;
		std::istringstream f(line);
		std::string s;
		while (getline(f, s, ':')) {
			s = trim(s);
			strings.push_back(s);
		}
		if (strings.size() >= 2)
		{
			if ((strings[0] == "PNG") && (strings[1] != "NO"))
			{
				opencv_formats["PNG"] = true;
			}
			else if ((strings[0] == "JPEG") && (strings[1] != "NO"))
			{
				opencv_formats["JPEG"] = true;
				opencv_formats["JPG"] = true;
				opencv_formats["JPE"] = true;
				opencv_formats["JP2"] = true;
			}
			else if ((strings[0] == "WEBP") && (strings[1] != "NO"))
			{
				opencv_formats["WEBP"] = true;
			}
			else if ((strings[0] == "TIFF") && (strings[1] != "NO"))
			{
				opencv_formats["TIFF"] = true;
			}
		}
	}
	// Windows Bitmaps (Always Supported)
	opencv_formats["BMP"] = true;
	opencv_formats["DIB"] = true;
	
	// Portable Image Format (Always Supported)
	opencv_formats["PBM"] = true;
	opencv_formats["PGM"] = true;
	opencv_formats["PPM"] = true;
	
	// Sun Rasters (Always Supported)
	opencv_formats["SR"] = true;
	opencv_formats["RAS"] = true;
}
void debug_show_opencv_formats()
{
	std::cout << "This is a list of supported formats, it depends on which formats opencv has been built with." << std::endl ;
	for (auto const& x : opencv_formats)
	{
		std::cout << x.first << " -> " << (x.second ? "Yes" : "No") << std::endl ;
	}
}

int main(int argc, char** argv) {
	int ret = 1;
	for (int ai = 1; ai < argc; ai++) {
		if (strcmp(argv[ai], "--list-processor") == 0) {
			dump_procs();
			return 0;
		}
		#ifdef HAVE_OPENCV
		if (strcmp(argv[ai], "--list-opencv-formats") == 0) {
			check_opencv_formats();
			debug_show_opencv_formats();
			return 0;
		}
		#endif
	}
	#ifdef HAVE_OPENCV
	check_opencv_formats();
	#endif

	// definition of command line arguments
	TCLAP::CmdLine cmd("waifu2x OpenCV Fork - https://github.com/DeadSix27/waifu2x-converter-cpp", ' ', _VERSION, true);
	cmd.setOutput(new CustomFailOutput());

	TCLAP::ValueArg<std::string> cmdInput("i", "input",
		"path to input image file or directory (you should use the full path)", true, "",
		"string", cmd);

	TCLAP::ValueArg<std::string> cmdOutput("o", "output",
		"path to output image file or directory  (you should use the full path)", false,
		"auto", "string", cmd);

	TCLAP::ValueArg<bool> cmdRecursiveDirectoryIterator("r", "recursive_directory",
		"Search recursively through directories to find more images to process. \n If this is set to 0 it will only check in the directory specified if the input is a directory instead of an image. \n You mustn't supply this argument with something other than 0 or 1.", false,
		0, "bool", cmd);


	TCLAP::SwitchArg cmdQuiet("q", "quiet", "Enable quiet mode.", cmd, false);

	std::vector<std::string> cmdModeConstraintV;
	cmdModeConstraintV.push_back("noise");
	cmdModeConstraintV.push_back("scale");
	cmdModeConstraintV.push_back("noise_scale");
	TCLAP::ValuesConstraint<std::string> cmdModeConstraint(cmdModeConstraintV);
	TCLAP::ValueArg<std::string> cmdMode("m", "mode", "image processing mode",
		false, "noise_scale", &cmdModeConstraint, cmd);

	std::vector<int> cmdNRLConstraintV;
	cmdNRLConstraintV.push_back(1);
	cmdNRLConstraintV.push_back(2);
	cmdNRLConstraintV.push_back(3);
	TCLAP::ValuesConstraint<int> cmdNRLConstraint(cmdNRLConstraintV);
	TCLAP::ValueArg<int> cmdNRLevel("", "noise_level", "noise reduction level",
		false, 1, &cmdNRLConstraint, cmd);

	TCLAP::ValueArg<double> cmdScaleRatio("", "scale_ratio",
		"custom scale ratio", false, 2.0, "double", cmd);

	TCLAP::ValueArg<std::string> cmdModelPath("", "model_dir",
		"path to custom model directory (don't append last / )", false,
		DEFAULT_MODELS_DIRECTORY, "string", cmd);

	TCLAP::ValueArg<int> cmdNumberOfJobs("j", "jobs",
		"number of threads launching at the same time", false, 0, "integer",
		cmd);

	TCLAP::ValueArg<int> cmdTargetProcessor("p", "processor",
		"set target processor", false, -1, "integer",
		cmd);

	TCLAP::SwitchArg cmdForceOpenCL("", "force-OpenCL",
		"force to use OpenCL on Intel Platform",
		cmd, false);

	TCLAP::SwitchArg cmdDisableGPU("", "disable-gpu", "disable GPU", cmd, false);

	TCLAP::ValueArg<int> cmdBlockSize("", "block_size", "block size",
		false, 0, "integer", cmd);
	TCLAP::SwitchArg cmdListProcessor("", "list-processor", "dump processor list", cmd, false);
	
	#ifdef HAVE_OPENCV
	TCLAP::SwitchArg showOpenCVFormats("", "list-opencv-formats", "dump opencv supported format list", cmd, false);
	#endif

	// definition of command line argument : end

	// parse command line arguments
	try {
		cmd.parse(argc, argv);
	}
	catch (std::exception &e) {
		std::cerr << e.what() << std::endl;
		std::cerr << "Error : cmd.parse() threw exception" << std::endl;
		std::exit(-1);
	}

	//We need to do this conversion because using a TCLAP::ValueArg<fs::path> can not handle spaces.
	fs::path input = cmdInput.getValue();
	std::string tmpOutput = cmdOutput.getValue();
	if (fs::is_directory(input) && (tmpOutput.back() != '/') && strcmp(tmpOutput.c_str(), "auto")!=0) {
		tmpOutput += "/";
	}
	fs::path output = tmpOutput;


	if (cmdListProcessor.getValue()) {
		dump_procs();
	}
	
	#ifdef HAVE_OPENCV
	if (showOpenCVFormats.getValue()) {
		debug_show_opencv_formats();
	}
	#endif
	
	enum W2XConvGPUMode gpu = W2XCONV_GPU_AUTO;

	if (cmdDisableGPU.getValue()) {
		gpu = W2XCONV_GPU_DISABLE;
	}
	else if (cmdForceOpenCL.getValue()) {
		gpu = W2XCONV_GPU_FORCE_OPENCL;
	}

	W2XConv *converter;
	int num_proc;
	w2xconv_get_processor_list(&num_proc);
	int proc = cmdTargetProcessor.getValue();
	bool verbose = !cmdQuiet.getValue();

	if (proc != -1 && proc < num_proc) {
		converter = w2xconv_init_with_processor(proc, cmdNumberOfJobs.getValue(), verbose);
	}
	else {
		converter = w2xconv_init(gpu, cmdNumberOfJobs.getValue(), verbose);
	}

	convInfo convInfo(cmdMode.getValue(), cmdNRLevel.getValue(), cmdScaleRatio.getValue(), cmdBlockSize.getValue(), converter);

	double time_start = getsec();

	switch (converter->target_processor->type) {
	case W2XCONV_PROC_HOST:
		printf("CPU: %s\n",
			converter->target_processor->dev_name);
		break;

	case W2XCONV_PROC_CUDA:
		printf("CUDA: %s\n",
			converter->target_processor->dev_name);
		break;

	case W2XCONV_PROC_OPENCL:
		printf("OpenCL: %s\n",
			converter->target_processor->dev_name);
		break;
	}

	bool recursive_directory_iterator = cmdRecursiveDirectoryIterator.getValue();
	int error = w2xconv_load_models(converter, cmdModelPath.getValue().c_str());
	check_for_errors(converter, error);

	//This includes errored files.
	int numFilesProcessed = 0;
	int numErrors = 0;
	if (fs::is_directory(input) == true) {
		//Build files list
		std::deque<fs::path> files_list;
		std::cout << "We're going to be operating in a directory. dir:" << fs::absolute(input) << std::endl;
		if (recursive_directory_iterator) {
			for (auto & inputFile : fs::recursive_directory_iterator(input)) {
				if (!fs::is_directory(inputFile)) {
					files_list.push_back(inputFile);
				}
			}
		}
		else {
			for (auto & inputFile : fs::directory_iterator(input)) {
				if (!fs::is_directory(inputFile)) {
					files_list.push_back(inputFile);
				}
			}
		}

		//Proceed by list
		double timeAvg = 0.0;
		int files_count = static_cast<int>(files_list.size());
		for (auto &fn : files_list) {
			++numFilesProcessed;
			double time_file_start = getsec();

			std::cout << "[" << numFilesProcessed << "/" << files_count << "] " << fn.filename() << (verbose ? "\n" : " Ok. ");

			try {
				convert_file(convInfo, fn, output);
			}
			catch (const std::exception& e) {
				numErrors++;
				std::cout << e.what() << std::endl;
			}

			//Calculate and out elapsed time
			double time_end = getsec();
			double time_file = time_end - time_file_start;
			double time_all = time_end - time_start;
			if (timeAvg > 0.0)
				timeAvg = time_all / numFilesProcessed;
			else
				timeAvg = time_all;
			double elapsed = files_count * timeAvg - time_all;
			int el_h = (int) elapsed / (60 * 60);
			int el_m = (int) (elapsed - el_h * 60 * 60) / 60;
			int el_s = (int) (elapsed - el_h * 60 * 60 - el_m * 60);
			std::cout << "Elapsed: ";
			if (el_h)
				std::cout << el_h << "h";
			if (el_m)
				std::cout << el_m << "m";
			std::cout << el_s << "s file: " << time_file << "s avg: " << timeAvg << "s" << std::endl;
		}


	}
	else {
		numFilesProcessed++;
		try {
			convert_file(convInfo, input, output);
		}
		catch (const std::exception& e) {
			numErrors++;
			std::cout << e.what() << std::endl;
		}
	}


	{
		double time_end = getsec();

		double gflops_proc = (converter->flops.flop / (1000.0*1000.0*1000.0)) / converter->flops.filter_sec;
		double gflops_all = (converter->flops.flop / (1000.0*1000.0*1000.0)) / (time_end - time_start);

		std::cout << "process successfully done! (all:"
			<< (time_end - time_start) << "[sec], "
			<< numFilesProcessed << " [files processed], "
			<< numErrors << " [files errored], "
			<< gflops_all << "[GFLOPS], filter:"
			<< converter->flops.filter_sec
			<< "[sec], " << gflops_proc << "[GFLOPS])" << std::endl;
	}

	w2xconv_fini(converter);

	return 0;
}
