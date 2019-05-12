/* Reference code for vector reduction. */

extern "C" double compute_gold(float *, const unsigned int);

double
compute_gold(float *input, const unsigned int len) 
{
	double sum = 0.0;
	
	for (unsigned int i = 0; i < len; i++)
		sum += input[i];

	return sum;
}

