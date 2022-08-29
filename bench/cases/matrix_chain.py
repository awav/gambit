import tensorflow as tf

def reduce_matrix_chain(A,B,C,D):
    return  tf.einsum('i,j->ij', tf.math.reduce_sum(A@B@C,axis=1), D)


def matrix_vector_chain(A,b,C,D):
  return tf.einsum('ij,j->i',D@tf.transpose(C), tf.einsum('ij,j->i', A, b))


def transpose_chain(A, B, C,D,E,F):
  part1 = tf.transpose(D@tf.transpose(E@F)) #10*30
  part2 = tf.transpose(A@B@C@part1)
  return part2

def pure_matrix_chain(A, B, C,D,E):
  return A@B@C@D@E