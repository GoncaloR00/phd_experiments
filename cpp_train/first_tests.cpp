#include <iostream>

int addNumbers(int first_param, int second_param){
    int result = first_param + second_param;
    return result;
}

int main(){
    for(size_t i=0; i<10000000; i++){
        int teste {5};
        int teste2 = 5;
        std::cout << "Hello world" << i << teste << teste2 << std::endl;
    }
    std::cout << addNumbers(10, 5) << std::endl;
    return 0;
}