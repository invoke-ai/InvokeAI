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
 * Disambiguate duplicate device names with backend-order 1-based suffixes across all installed accelerators,
 * including disabled ones.
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

/** Show badges only for reported indexed accelerators when multiple accelerators exist. */
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
