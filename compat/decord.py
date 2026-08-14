"""Compatibility stub for EchoMimicV3's unused optional decord import.

The DGX Spark aarch64 index does not publish a decord wheel. The current
EchoMimicV3 flash inference module imports decord but never calls it, so an
empty module keeps the supported single-GPU path portable without pretending
to implement video decoding.
"""
