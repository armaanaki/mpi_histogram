#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <sys/stat.h>

void handle_error (const char* error_message);
// precondition: error_message is not null
// postcondition: error_message has been displayed on standard error &
//                program has been terminated with an error code of -1
//                precondition has been verified with an assertion

double* read_from_file (const char* file_name, size_t file_size);
// precondition: file_name is not null - verified with assertion
// return: array contaiing file_size elements
// postcondition: handle_error has been called if the file does not exist
//                or error reading from file

size_t get_file_size (const char* file_name);
// precondition: file_name is not null - verified with assertion 
// returned: number of bytes in the file
// postcondition: handle_error has been called for any file errors

/************************************************************************/

void handle_error (const char* error_message)
{
  assert (error_message != NULL && "NULL string argument to handle_error");
  fprintf (stderr, error_message);
  fprintf (stderr, "\n");
  exit (-1);
}

/************************************************************************/

double* read_from_file (const char* file_name, size_t file_size)
{
  assert (file_name != NULL && "null file name");
  size_t num_doubles = get_file_size (file_name)/sizeof(double);
  if (num_doubles != file_size)
    handle_error ("data file is the wrong size");
  double* a = new double[num_doubles];
  FILE* f = fopen (file_name, "rb");
  if (f == NULL)
    handle_error ("error opening file");
  size_t amount = fread (a, sizeof(double), num_doubles, f);
  if (amount != num_doubles)
    handle_error("error in reading from file");
  fclose (f);
  return a;
}

/**************************************************************************/

size_t get_file_size (const char* file_name)
{
  assert (file_name != NULL && "null file name");
  struct stat st;
  stat(file_name, &st);
  return st.st_size;
}
