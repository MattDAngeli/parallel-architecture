/* Reference code for vector reduction. */

extern "C" double compute_gold (float *, unsigned int);


double 
compute_gold (float* A, unsigned int num_elements)
{
    unsigned int i;
    double sum = 0.0; 
  
    for (i = 0; i < num_elements; i++) 
        sum += A[i];
  
    return sum;
}







