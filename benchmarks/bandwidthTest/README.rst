Bandwidth Test
==============

This example evaluates the bandwidth between the host and the device (WSE). The
kernel records the ``start`` and ``end`` of H2D or D2H by tsc counter. This is
better than host timer because the runtime may not send the command right after
the user issues it. The runtime can aggregate multiple nonblocking commands
together to reduce TCP overhead. In addition the tsc counters of all PEs are
not sychronized in the beginning. To avoid the timing variation among those PEs
, we add a sync() to synchronize all PEs and sample the reference clock.

The kernel ``bw_sync_kernel.csl`` defines a couple of host-callable functions,
``f_sync()``, ``f_tic()`` and ``f_toc()`` in order to synchronize the PEs and
record the timing of H2D or D2H.

The kernel ``sync/pe.csl`` performs a reduction over the whole rectangle to sync
the PEs, then the top-left PE sends a signal to other PEs to sample the
reference clock.

The script ``run.py`` has the following parameters:

- ``--loop_count=<int>`` decides how many H2Ds/D2Hs are called.

- ``--d2h`` measures the bandwidth of D2H, otherwise H2D is measured.

- ``--channels=<int>`` specifies the number of I/O channels, no bigger than 16.

The tic() samples "time_start" and toc() samples "time_end". The sync() samples
"time_ref" which is used to adjust "time_start" and "time_end".
The elapsed time (unit: cycles) is measured by
``cycles_send = max(time_end) - min(time_start)``

The overall runtime (us) is computed via the following formula
``time_send = (cycles_send / 0.85) * 1.e-3 us``

The bandwidth is calculated by
``bandwidth = ((wvlts * 4)/time_send)*loop_count``
