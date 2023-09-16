
FIFOs
=====

A FIFO DSD is useful to buffer input going into or out of a PE, as a way to
extend the small hardware queues used for fabric communication. In particular,
this may prevent stalls in the communication fabric when input or output
happens in bursts. It is also possible to operate on the values while they flow
through the FIFO, as this code sample demonstrates.

This example illustrates a typical pattern in the use of FIFOs, where a
receiver receives wavelets from the fabric and forwards them to a task that
performs some computation. Specifically, incoming data from the host is stored
in the FIFO, thus relieving the sender from being blocked until the receiver
has received all wavelets. While the incoming wavelets are being asynchronously
received into the FIFO buffer, we also start a second asynchronous DSD
operation that pulls data from the FIFO and forwards it to a wavelet-triggered
task.

This example also illustrates another common pattern, where a PE starts a
wavelet-triggered task using its own wavelets, by sending them to the router
which immediately sends them back to the compute element. In our example, this
wavelet-triggered task simply computes the cube of the wavelet's data, before
sending the result to the host.
