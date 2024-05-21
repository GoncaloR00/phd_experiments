// MyClass.cpp
#include "MyClass.h"
#include <iostream>

void MyClass::publicMethod() {
    std::cout << "This is a public method.\n";
    privateMethod();
}

void MyClass::privateMethod() {
    std::cout << "This is a private method.\n";
}