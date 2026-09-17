#!/bin/bash
# Runs on a board, detached: build llama.cpp's library (the runtime links
# libllama for the tokenizer, the GGUF reader and the chat template).
export PATH=$HOME/cmake-4.4.3-linux-aarch64/bin:$PATH
cd ~
[ -d cmake-4.4.3-linux-aarch64 ] || tar xzf cmake-aarch64.tar.gz
[ -d llama.cpp ] || tar xzf llamacpp-src.tar.gz
cd ~/llama.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF > ~/llama-build.log 2>&1 && \
cmake --build build --target llama -j3 >> ~/llama-build.log 2>&1
echo "exit=$?" >> ~/llama-build.log
