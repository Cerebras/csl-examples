Topic 13: WSE-3 Microthreads
============================

Unlike WSE-2, the WSE-3 architecture exposes microthread IDs.
This example demonstrates the use of explicit microthread IDS
on the WSE-3 architecture.

On WSE-2, the queue ID of an input or output fabric DSD corresponds to the
ID of the microthread in which that operation executes.
On WSE-3, queue IDs and microthreads can be decoupled, so that any
microthread ID 0 to 7 can be used with any of queues 0 to 7.

In this example, the left PE sends ``M`` wavelets to the right PE over
the color ``send_color``.
These wavelets are sent in an asynchronous ``@fmovs`` operation which
copies from the ``y`` array via ``y_dsd`` into ``out_dsd``.
``out_dsd`` is a ``fabout_dsd`` associated with the color ``send_color``,
and the output queue with ID 2.
The ``@fmovs`` operation is launched using microthread ID 4.

The right PE receives these ``M`` wavelets on the same color (called
``right_color`` in ``right_pe.csl``) via ``in_dsd``, which uses input
queue with ID 2.
The asynchronous ``@fmovs`` operation which receives these wavelets
and copies them into ``y`` is launched using microthread ID 5.

Decoupling microthread IDs from queue IDs can provide valuable flexibility
in managing program resource usage, and conserve microthreads.

By using explicit microthread IDs, we allow CSL's DSR allocator to use fewer
DSRs in situations where fabric DSD operands are not known at compile time.

Additionally, on the WSE-3, output queues cannot be re-used with a different
color if they have not yet been drained, and CSL does not yet support a
mechanism for guaranteeing that a given queue is empty.
This may force the programmer to use more output queues than needed, which in
turn can lead to overusing microthread IDs (if they are not explicitly
specified, they default to the respective queue IDs).
By allowing explicit microthread IDs, a programmer can share microthreads
between output queues, and thus conserve microthreads for other operations.
Note, however, that two operations cannot concurrently use the same microthread.
