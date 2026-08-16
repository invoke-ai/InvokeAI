/**
 * Labels for the accelerator a queue item ran on.
 *
 * With `generation_devices` (default `auto`) the backend runs one session per accelerator
 * concurrently, and tags each queue item and progress event with the device that
 * processed it. These helpers turn `"cuda:1"` into something a person can read.
 */

/** One device the backend offers for generation, from `GET /api/v1/app/generation_device_options`. */
export interface GenerationDeviceOption {
  /** Device identifier, e.g. `cuda:0`, `xpu:0`, `mps`, `cpu`. */
  device: string;
  /** Human-readable device name, e.g. `NVIDIA GeForce RTX 5090`. */
  name: string;
}

/**
 * Parse the accelerator index out of a device string (`"cuda:1"` / `"xpu:1"` → `1`).
 *
 * Returns null for a missing or unindexed device (`"cpu"`, `"mps"`).
 */
export const getCudaDeviceIndex = (device: string | null | undefined): number | null => {
  if (!device) {
    return null;
  }

  const match = /^(?:cuda|xpu):(\d+)$/.exec(device);

  return match ? Number(match[1]) : null;
};

/** Resolve descriptive random-device metadata without leaking the device store into graph compilers. */
export const resolveRandDeviceMetadata = (useCpuNoise: boolean, options: readonly GenerationDeviceOption[]): string => {
  if (useCpuNoise) {
    return 'cpu';
  }

  const types = new Set(options.map(({ device }) => device.split(':')[0]).filter((type) => type !== 'cpu'));
  const [deviceType] = types;

  return types.size === 1 && deviceType ? deviceType : 'cuda';
};

/**
 * Map device id → display label, disambiguating identically-named GPUs with a
 * 1-based `#N` suffix. A uniquely-named device gets no suffix.
 *
 * The ordinal follows the order the options are given, and the backend returns
 * every installed CUDA device in index order — not just the ones enabled in
 * `generation_devices`. That matters: numbering off a filtered list would renumber
 * the survivors when a device is disabled, so `cuda:2` would stop being `#3`.
 */
export const getDeviceNameLabels = (options: readonly GenerationDeviceOption[]): Record<string, string> => {
  const nameCounts = new Map<string, number>();

  for (const option of options) {
    nameCounts.set(option.name, (nameCounts.get(option.name) ?? 0) + 1);
  }

  const ordinals = new Map<string, number>();
  const labels: Record<string, string> = {};

  for (const option of options) {
    const ordinal = (ordinals.get(option.name) ?? 0) + 1;

    ordinals.set(option.name, ordinal);
    labels[option.device] = (nameCounts.get(option.name) ?? 0) > 1 ? `${option.name} #${ordinal}` : option.name;
  }

  return labels;
};

export interface DeviceLabel {
  /** Accelerator index, for the compact badge (e.g. `0`). */
  index: number;
  /** Full name with disambiguating ordinal, for the tooltip (e.g. `NVIDIA GeForce RTX 5090 #1`). */
  name: string;
}

/**
 * Resolve a device string to the badge + tooltip pair, or null when there is
 * nothing worth showing.
 *
 * Null covers three cases: the device is not an indexed accelerator, only one accelerator exists (no
 * ambiguity to resolve, so a badge would be noise), or the device is absent from
 * the reported options.
 */
export const getDeviceLabel = (
  device: string | null | undefined,
  options: readonly GenerationDeviceOption[]
): DeviceLabel | null => {
  const index = getCudaDeviceIndex(device);

  if (index === null || options.length <= 1 || !device) {
    return null;
  }

  const name = getDeviceNameLabels(options)[device];

  return name ? { index, name } : null;
};
