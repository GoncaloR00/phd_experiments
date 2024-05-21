#include <iostream>


int main(){
    const int scores[10] {0, 0, 0, 0, 0, 0 ,0 ,0, 0 ,0}; //Const locks value in memory

    // This way we lost the index
    for (auto score:scores){
        std::cout << "score: " << score << std::endl;
    }

    int size{std::size(scores)};
    // or
    int size2{sizeof(scores)/sizeof(scores[0])};
    for (size_t i{0};i< size; i++){ // size_t = unsigned int but more commonly used in loops
        std::cout << "score " << i <<": " <<scores[i] << std::endl;
    }
    for (unsigned int i{0};i< size2; i++){
        std::cout << "score " << i <<": " <<scores[i] << std::endl;
    }
    std::cerr << "error "<< std::endl;
    std::cout << "end "<< std::endl;
    return 0;
}