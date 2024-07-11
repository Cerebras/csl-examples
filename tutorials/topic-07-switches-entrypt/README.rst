Topic 7: Switches and Control Entrypoints
=========================================

Following on from the last topic example, we can also encode a special
task ID inside of a control wavelet. When that control wavelet is forwarded
to the CE of the receiving PE, it will activate a task known as a control
task which is bound to that ID.

The lower 16 bits of the control wavelet can be used to store an optional
data payload for that control task. Here, we encode the same values
sent to the PEs as normal data wavelets in the previous example.

Note that a PE router will move to a new switch position only after the
control wavelet carrying the switch command passes through that PE.
Therefore all control wavelets will continue to be routed using the current
switch position setting and the new switch position will only affect
subsequent wavelets. Thus, the data payload of a control wavelet is received
by the PE connected by the current switch position, not the new position.
