#include <vector>
#include <map>
#include <fstream>
#include <iostream>

#define __CL_ENABLE_EXCEPTIONS 
#include <CL/cl.hpp>
#include <CL/cl.h>

using namespace std;

class EasyOpenCL {
private:
	cl::Platform default_platform;
	cl::Device default_device;

	cl::Context context;
	cl::CommandQueue queue;

	map<string, cl::Kernel> kernels;
public:
	EasyOpenCL() {
		init();
	}
	void init() {
		try {
			cout << "Initializing OpenCL..." << endl;
			//get all platforms (drivers)
			vector<cl::Platform> all_platforms;
			cl::Platform::get(&all_platforms);
			if (all_platforms.size() == 0) {
				cout << " No platforms found. Check OpenCL installation!\n";
				exit(1);
			}
			default_platform = all_platforms[0];
			cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

			//get default device of the default platform
			vector<cl::Device> all_devices;
			default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
			if (all_devices.size() == 0) {
				cout << " No devices found. Check OpenCL installation!\n";
				exit(1);
			}
			default_device = all_devices[0];
			cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

			// create context
			context = cl::Context({ default_device });

			//create queue to which we will push commands for the device.
			queue = cl::CommandQueue(context, default_device);
		}
		catch (cl::Error e) {
			cout << endl << "a OpenCL-Error occured while initializing:" << endl;
			cout << e.what() << " : " << e.err() << endl;
		}
	}

	void createNewKernel(const char* kernel_name) {
		try {
			//load kernel as string
			ifstream cl_file(string(kernel_name)+".cl");
			string cl_string(istreambuf_iterator<char>(cl_file), (istreambuf_iterator<char>()));
			cl::Program::Sources source(1, make_pair(cl_string.c_str(), cl_string.length() + 1));

			// create program
			cl::Program program(context, source);

			// compile opencl source
			if (program.build({ default_device }) != CL_SUCCESS) {
				cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
				exit(1);
			}

			// load named kernel from opencl source
			cl::Kernel kernel(program, kernel_name);
			kernels[string(kernel_name)] = kernel;
		}
		catch (cl::Error e) {
			cout << endl << "a OpenCL-Error occured while creating context:" << endl;
			cout << e.what() << " : " << e.err() << endl;
		}
	}

	void run_example() {
		try {
			const char * kernel_name = "simple_add";

			createNewKernel(kernel_name);
			cl::Kernel kernel = kernels[string(kernel_name)];

			// create buffers on the device
			int A[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
			int B[] = { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };

			cout << endl << "A Simple Example that adds two seperate int arrayss:";
			cout << endl << "input 1: \n\t";
			for (int i = 0; i < 10; i++) {
				cout << A[i] << " ";
			}
			cout << endl << "input 2: \n\t";
			for (int i = 0; i < 10; i++) {
				cout << B[i] << " ";
			}

			cl::Buffer buffer_A(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 10, A);
			cl::Buffer buffer_B(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * 10, B);
			cl::Buffer buffer_C(context, CL_MEM_WRITE_ONLY, sizeof(int) * 10);

			kernel.setArg(0, buffer_A);
			kernel.setArg(1, buffer_B);
			kernel.setArg(2, buffer_C);

			// execute kernel
			cl::NDRange global_work_size = cl::NDRange(10);
			queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_work_size, cl::NullRange);

			// wait for completion
			queue.finish();

			//read result C from the device to array C
			int C[10];
			queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(int) * 10, C);

			cout << endl << "result: \n\t";
			for (int i = 0; i < 10; i++) {
				cout << C[i] << " ";
			}
		}
		catch (cl::Error e) {
			cout << endl << "a OpenCL-Error occured while Enqueuing Buffers:" << endl;
			cout << e.what() << " : " << e.err() << endl;
		}
	}
};

int main() {
	EasyOpenCL open_cl;

	open_cl.run_example();

	cin.get();
	return 0;
}