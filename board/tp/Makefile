# Built on a board. Needs ~/llama.cpp built (libllama, libggml, libggml-base)
# and ~/board/kv260gemv (the engine driver).
LLAMA   ?= $(HOME)/llama.cpp
GEMV    ?= $(HOME)/board/kv260gemv
CXX     ?= g++
CC      ?= gcc
CXXFLAGS ?= -O2 -std=c++17 -Wall -pthread -I$(LLAMA)/include -I$(LLAMA)/ggml/include -I$(LLAMA)/ggml/src -I$(LLAMA)/vendor -I$(LLAMA)/vendor/cpp-httplib -I$(GEMV)
CFLAGS  ?= -O2 -std=gnu11 -I$(GEMV)
LDFLAGS ?= -L$(LLAMA)/build/bin -Wl,-rpath,$(LLAMA)/build/bin -lllama -lggml -lggml-base -lm -pthread

tp_llama: tp_llama.o httplib.o udmabuf.o gemv_dev.o gemv2_dev.o
	$(CXX) -o $@ $^ $(LDFLAGS)

tp_llama.o: tp_llama.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# cpp-httplib ships split in llama.cpp's vendor dir: header plus this implementation file.
httplib.o: $(LLAMA)/vendor/cpp-httplib/httplib.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

udmabuf.o: $(GEMV)/udmabuf.c
	$(CC) $(CFLAGS) -c -o $@ $<

gemv_dev.o: $(GEMV)/gemv_dev.c
	$(CC) $(CFLAGS) -c -o $@ $<

gemv2_dev.o: $(GEMV)/gemv2_dev.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o tp_llama
