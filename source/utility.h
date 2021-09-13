#ifndef _UTILITY_H
#define _UTILITY_H

#include <string>
#include <stdio.h>
#include <stdarg.h> 
#include <time.h>

inline std::string cStringFormatA(const char * fmt, ...)
{
    int nSize = 0;
    char buff[4096];
    va_list args;
    va_start(args, fmt);
    nSize = vsnprintf(buff, sizeof(buff) - 1, fmt, args); // C4996
    return std::string(buff);
}


#endif