Topic 11: Collective Communications
===================================

The ``<collectives_2d>`` library can be used for communication between PEs in
the same row or column. It mimics the capabilities provided by
`message passing interface <https://www.open-mpi.org/>`_ (MPI)
collective operations found in other programming languages.

This example showcases each of the currently available communication primitives
while using the library across two indepedent dimensions. The communication
tasks are executed asynchronously.

``task_x`` uses the ``broadcast`` primitive to transmit data from the first PE
in every row to every other PE in the same row. After the data is received,
``reduce_fadds`` computes the vector sum of the ``broadcast_recv``. The result
is transmitted back to the first PE in every row.

``task_y`` operates concurrently along every column of PEs. The task first
uses ``scatter`` to distribute ``chunk_size`` slices of ``scatter_data``
across every other PE in the same column. The task uses ``gather`` to collect
``chunk_size`` slices of data distributed by ``scatter``. Because ``scatter``
is the inversion of ``gather``, we have used collective communications to
transmit the data from ``scatter_data`` to ``gather_recv``.
