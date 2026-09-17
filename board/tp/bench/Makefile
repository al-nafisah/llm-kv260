# Microbenchmarks of the runtime's CPU path, built and run on a board
# (buildlog/0010). ops.inc is the ops section of tp_llama.cpp, extracted,
# so ops_bench and attn_var time the kernels exactly as committed.
LLAMA ?= $(HOME)/llama.cpp
CXXFLAGS ?= -O2 -std=c++17 -I$(LLAMA)/ggml/include
LDFLAGS ?= -L$(LLAMA)/build/bin -Wl,-rpath,$(LLAMA)/build/bin -lggml-base -lm -pthread
all: attn_bench misc_bench ops_bench attn_var mesh_bench
ops.inc: ../tp_llama.cpp
	sed -n '/^\/\/ -* ops$$/,/^\/\/ -* stats$$/p' $< | sed '$$d' > $@
comm.inc: ../tp_llama.cpp
	sed -n '/^\/\/ -* comm$$/,/^\/\/ -* ops$$/p' $< | sed '$$d' > $@
mesh_bench: mesh_bench.cpp comm.inc
	$(CXX) $(CXXFLAGS) -o $@ $<
ops_bench attn_var: %: %.cpp ops.inc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
misc_bench: misc_bench.cpp
	$(CXX) $(CXXFLAGS) -I$(LLAMA)/ggml/src -o $@ $< $(LDFLAGS) -lggml
attn_bench: attn_bench.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -pthread
run: all
	for p in 40 200 400 1000; do ./attn_bench $$p; done; ./misc_bench; ./ops_bench 2048; for p in 200 400 1000; do ./attn_var $$p; done
clean:
	rm -f attn_bench misc_bench ops_bench attn_var mesh_bench ops.inc comm.inc
