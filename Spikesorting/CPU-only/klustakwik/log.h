#ifndef LOG_H_
#define LOG_H_

#include <stdio.h>

extern FILE *logfp;

void Output(char *fmt, ...);
void Error(char *fmt, ...);
void FlushLog();

#endif
