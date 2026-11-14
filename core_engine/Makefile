CXX ?= clang++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O3 -fPIC -Iinclude -Ithird_party/llama.cpp/include -Ithird_party/llama.cpp/ggml/include -I/data/user/0/com.termux/files/usr/include -DSTREAMVEX_BUILDING_DLL -pthread -mcpu=native -fopenmp
LDFLAGS ?= -shared -pthread -fopenmp

TARGET = libstreamvex.so
SRCS = src/streamvex.cpp \
       src/jni_bridge.cpp \
       src/io/layer_streamer.cpp \
       src/parser/gguf_reader.cpp \
       src/compute/dequant.cpp \
       src/compute/gemv.cpp \
       src/compute/transformer_layer.cpp \
       src/compute/sampler.cpp

LLAMA_OBJS = third_party/llama.cpp/src/llama.o \
             third_party/llama.cpp/src/llama-vocab.o \
             third_party/llama.cpp/src/llama-grammar.o \
             third_party/llama.cpp/src/llama-sampling.o \
             third_party/llama.cpp/src/unicode.o \
             third_party/llama.cpp/src/unicode-data.o \
             third_party/llama.cpp/ggml/src/ggml.o \
             third_party/llama.cpp/ggml/src/ggml-alloc.o \
             third_party/llama.cpp/ggml/src/ggml-backend.o \
             third_party/llama.cpp/ggml/src/ggml-quants.o \
             third_party/llama.cpp/ggml/src/ggml-aarch64.o \
             third_party/llama.cpp/ggml/src/llamafile/sgemm.o

OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LLAMA_OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
