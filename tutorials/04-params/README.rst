
Parameters
==========

This example shows a slightly sophisticated use of parameters, wherein the code
switches one type for another based on the parameter value.

Parameter values are compile-time constants, which implies that the compiler is
fully aware of their precise value.  This enables the programmer to not just
change the program's behavior at runtime, but it also enables the programmer to
change the program's compilation.

This example defines a function ``fetchType()``, whose return type is ``type``,
which is the type of all types.  Subsequently, when the code defines a new
variable (``global``), it calls the ``fetchType()`` function to decide the type of
the variable.

While this code uses an integer parameter, the programmer is free to declare
parameters of other types such as floats, bools, or arrays.  However, the
command-line interface for initializing parameters supports signed integers
only.
