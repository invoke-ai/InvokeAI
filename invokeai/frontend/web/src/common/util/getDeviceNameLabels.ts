import type { S } from 'services/api/types';

/**
 * Build a map of device id (e.g. `"cuda:0"`) → human-readable label (e.g. `"AMD Radeon PRO W7900 #1"`).
 *
 * Devices that share a name get a 1-based `#N` suffix so identical GPUs can be told apart; a
 * uniquely-named device gets no suffix. The ordinal is assigned in the order the options are
 * provided (which the backend returns in CUDA-index order). Used to label progress previews with
 * the GPU they are rendering on in multi-GPU setups.
 */
export const getDeviceNameLabels = (options: S['GenerationDeviceOption'][]): Record<string, string> => {
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
