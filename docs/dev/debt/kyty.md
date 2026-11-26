# kyty technical debt

- [Kyty dynamic vertex fetch](../../../core/kyty/graphics/shader/shader.cpp) still rejects a live vertex-buffer descriptor with `ADD_TID_ENABLE`. The fixed-function Vulkan vertex input carries base, stride and vertex or instance rate. Supporting the flag requires reproducing lane-dependent indexing and bounds checks. An error during speculative vertex preparation falls back to ordinary execution of the packet.
