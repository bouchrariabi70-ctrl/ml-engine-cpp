// CVSReader.h
#ifndef CVSREADER_H
#define CVSREADER_H

#include "Matrix.h"
#include "Vector.h"
#include <string>

class CVSReader {
public:
    static Matrix lireMatrix(const std::string& chemin,
                              bool ignoreHeader = true);
    static Vector extraireColonne(const Matrix& data, int col);
    static Matrix extraireFeatures(const Matrix& data);
};

#endif