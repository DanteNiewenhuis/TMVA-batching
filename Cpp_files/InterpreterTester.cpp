#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>

// Include ROOT files
#include "TMVA/RTensor.hxx"
#include "ROOT/RDataFrame.hxx"

#include "TInterpreter.h"

#include <chrono>
#include <fstream>


void InterpreterTester()
{
    gInterpreter->Declare("auto x = 5 + 8;");

    std::cout << x << std::endl;
}