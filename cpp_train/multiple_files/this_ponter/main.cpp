#include <iostream>

class MyClass {
public:
    int x {10};
    MyClass* pointer_x(){
        std::cout<<"inside class x pointer: " << &x << std::endl;
        return this;
    }
    MyClass* setX(int x) {// MyClass* porque estamos a retornar um pointer para o objeto (objeto é a variável criada para construir a class)
        std::cout<<"x is: " << x << std::endl;
        std::cout<<"this x is: " << this->x << std::endl;
        std::cout<<"this x pointer: " << this << std::endl;
        this->x = x;  // 'this' pointer distinguishes between the parameter 'x' and the data member 'x'
        return this;

    }
};

int main(){
    MyClass teste;
    std::cout<<"class pointer is: " << &teste << std::endl;
    std::cout<<"x class pointer is: " << &teste.x << std::endl;
    teste.setX(100);
    teste.pointer_x();
    // Como fizemos "return this", podemos fazer de outra forma:
    teste.setX(1000)->pointer_x();
    return 0;
}