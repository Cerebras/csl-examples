
Asynchronous Operations
=======================

Each PE supports execution in several threads, each of which runs concurrently.
However, the execution of code in threads can only be triggered using DSD
operations.  This example ``mov16`` operation, which is programmed to run in a
separate thread using the ``.async = true`` field.

Since threads run concurrently, the programmer may be interested in knowing when
a thread has finished execution.  To that end, this example tells the compiler
to generate code to unblock the ``thisTask`` when the thread has finished
execution.  Similar to unblocking, tasks can also be *activated* when a thread
has terminated.

Although different threads execute concurrently, tasks always run in a special
thread, called the main thread.  Consequently, task execution is serialized,
even though asynchronous DSD operations are not serialized.
