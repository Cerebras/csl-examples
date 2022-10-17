
Loops
=====

The generic version of CSL's ``while`` loops, shown in previous examples, is
similar to the one in many other languages.  In CSL, ``while`` loops can also
have a ``continue`` expression, which contains statements that must execute at
the end of the each iteration.  Such ``continue`` expressions are useful to
ensure that we don't accidentally forget to, say, increment loop counters.

The ``for`` loop syntax in CSL is somewhat different from that in many other
languages.  Specifically, CSL's ``for`` loop accepts a single argument, which
is the array to iterate on, which is followed by one or two values between ``|``
characters.  If only one name is specified, like ``pe_x`` in the example, this
name is a ``const`` variable whose value is set to each element of the array.
If two names are specified, the second name is another ``const`` variable, the
value of which is set to the index of the array element.
