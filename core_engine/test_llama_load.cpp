#include <iostream>
#include "third_party/llama.cpp/include/llama.h"

int main() {
    llama_backend_init();
    std::cout << "llama_backend_init OK" << std::endl;
    llama_backend_free();
    return 0;
}
