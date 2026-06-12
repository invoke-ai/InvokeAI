/**
 * Parse the CUDA device index from a device string (e.g. `"cuda:1"` → `1`).
 *
 * Returns `null` when the device is null/undefined or is not a CUDA device (e.g. `"cpu"`, `"mps"`).
 * Used to label progress previews and queue items with their GPU number in multi-GPU setups.
 */
export const getCudaDeviceIndex = (device: string | null | undefined): number | null => {
  if (!device) {
    return null;
  }
  const match = /^cuda:(\d+)$/.exec(device);
  return match ? Number(match[1]) : null;
};
