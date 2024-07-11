Topic 8: Filters
================

Fabric filters allow a PE to selectively accept incoming wavelets.  This example
shows the use of so-called range filters, which specify the wavelets to allow to
be forwarded to the CE based on the upper 16 bits of the wavelet contents.
Specifically, PE #0 sends all 12 wavelets to the other PEs, while each recipient
PE receives and processes only a quarter of the incoming wavelets.
See :ref:`language-builtins-filters` for other possible filter configurations.
