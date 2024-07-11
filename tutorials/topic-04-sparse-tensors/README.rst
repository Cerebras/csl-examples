Topic 4: Wavelets for Sparse Tensors
====================================

When tensors are sparse, it is wasteful to send zero values.  Since wavelet
payloads are 32 bits wide, we can use the lower 16 bits to contain data as
usual, but we can also use the upper 16 bits to contain the index of the value.

This example illustrates the latter, where each wavelet of the incoming tensor
has the index field populated in the upper 16 bits.  Accordingly, the task
definition uses two function arguments, one for the lower 16 bits whereas
another for the upper 16 bits.

Optionally, the programmer may also declare a task with just one argument of
type ``u32`` for receiving 32-bit data.
