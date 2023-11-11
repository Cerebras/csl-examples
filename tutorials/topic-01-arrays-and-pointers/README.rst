
Topic 1: Arrays and Pointers
============================

Arrays can only be passed to or returned from functions used at compile-time.
For functions used at runtime, pointers should be used instead.  This example
demonstrates a function ``incrementAndSum()``, which accepts a pointer to an
array and a pointer to a scalar.  When declaring an array pointer, CSL requires
that the type specification contain the size of the array.  CSL does not have
a null pointer.

Pointers are dereferenced using the ``.*`` syntax.  Once dereferenced, they can
be used just like non-pointer variables like ``(dataPtr.*)[0]`` for indexing
into the first element of the array.
