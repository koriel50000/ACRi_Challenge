#include "kernel.hpp"
#include "image.hpp"
#include "params.hpp"

void input_stream(hls::stream<uint64_t>& ins) {
    for (int i = 0; i < 640 * 640; i++) {
        ins.write(0);
    }
}

int main(int argc, char** argv)
{
    hls::stream<uint64_t> ins;
	int out[16];

    input_stream(ins);
	kernel(ins, out);
    printf("out[0]=%d\n", out[0]);

	return EXIT_FAILURE; //EXIT_SUCCESS;
}
