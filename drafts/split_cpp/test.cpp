#include <stdio.h>
#define int64 long long int

const int64 primes[64] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
                          37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
                          83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
                          139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
                          197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257,
                          263, 269, 271, 277, 281, 283, 293, 307, 311};

int64 BestSplitSizeFold(int64 (&factors)[64], int offset, int64 current,
                        int64 best, int64 size, int64 max_size)
{
  if (offset >= 64)
  {
    return best;
  }
  else
  {
    if (factors[offset] > 0)
    {
      int64 current_prime = primes[offset] * current;
      printf("split: %li\n", current_prime);
      if (size / current_prime <= max_size && current_prime < best)
      {
        best = current_prime;
      }
      factors[offset]--;
      best = BestSplitSizeFold(factors, offset, current_prime, best, size, max_size);
      factors[offset]++;
    }
    return BestSplitSizeFold(factors, offset + 1, current, best, size, max_size);
  }
}

int64 BestSplitSize(int64 size, int64 full_size_bytes, int64 max_intermediate_bytes)
{
  // find list of prime factors
  int64 factors[64];
  int64 tmp_size = size; // inst->shape().dimensions(split_dim);
  for (int i = 0; i < 64; i++)
  {
    factors[i] = 0;
    while (tmp_size % primes[i] == 0)
    {
      factors[i]++;
      tmp_size /= primes[i];
    }
  }

  // int64 size = inst->shape().dimensions(split_dim);
  // int64 full_size_bytes =
  //     ShapeUtil::ByteSizeOfPrimitiveType(inst->shape().element_type()) *
  //     ShapeUtil::ElementsIn(inst->shape());
  int64 max_size = max_intermediate_bytes * size / full_size_bytes;
  printf("max_size: %li\n", max_size);
  return BestSplitSizeFold(factors, 0, 1, size, size, max_size);
}

int main(int argc, char *argv[])
{
  int64 size = 100000;
  int64 full_size_bytes = size * size * (64 / 8);
  int64 max_intermediate_bytes = 15 * 134217728;

  int64 split = BestSplitSize(size, full_size_bytes, max_intermediate_bytes);

  printf(
      "BestSplitSize(%li, %li, %li) = %li (for slices of %li)\n",
      size, full_size_bytes, max_intermediate_bytes, split, size / split);
  return 0;
}
