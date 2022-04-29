
Modules
#######

For the sake of a cleaner organization of source files, CSL permits the use
of modules, which are essentially CSL source files that can be imported into
other source files.

This example creates a module for producing numbers in the Fibonacci sequence.
The module is parameterized with the maximum numbers that the code may be asked
to produce.

Modules are described in greater detail `here <../../Language/Modules.rst>`_.
Modules can be recursively imported and with proper ``comptime`` logic, such
recursive imports *can* be safe, but if you use recursive imports, tread
carefully!
