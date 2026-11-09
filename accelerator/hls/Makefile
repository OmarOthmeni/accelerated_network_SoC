# C simulation of the HLS kernel with plain g++ (no Xilinx tools needed).
#   make csim        build + run the bit-exactness test on the whole test set
# On the PYNQ board, the same target gives the ARM Cortex-A9 software baseline:
#   make csim CXXFLAGS="-O3 -mcpu=cortex-a9 -mfpu=neon"   (time it with `time ./tb`)

CXX      ?= g++
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++14 -Wall -Wno-unknown-pragmas -Wno-unused-label -I. -DDATA_DIR=\"data\"

.PHONY: csim clean
csim: tb
	./tb

tb: tb.cpp cnn.cpp cnn.h weights.h
	$(CXX) $(CXXFLAGS) tb.cpp cnn.cpp -o tb

clean:
	rm -f tb
