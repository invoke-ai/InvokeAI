import { getCudaDeviceIndex } from 'common/util/getCudaDeviceIndex';
import { getDeviceNameLabels } from 'common/util/getDeviceNameLabels';
import { useMemo } from 'react';
import { useGetGenerationDeviceOptionsQuery } from 'services/api/endpoints/appInfo';

type ProgressDeviceLabel = {
  /** The CUDA device index, shown in the center of the progress circle (e.g. `0`). */
  index: number;
  /** The human-readable device name and number, shown on hover (e.g. `"AMD Radeon PRO W7900 #1"`). */
  name: string;
};

/**
 * Resolve a device string (e.g. `"cuda:0"`) to the GPU index + name used to annotate progress
 * previews.
 *
 * Returns `null` when there is nothing to show: the device is not a CUDA GPU, or only a single GPU
 * is available (single-GPU setups show neither the index nor the hover tooltip).
 */
export const useProgressDeviceLabel = (device: string | null | undefined): ProgressDeviceLabel | null => {
  const { data: deviceOptions } = useGetGenerationDeviceOptionsQuery();

  return useMemo(() => {
    const index = getCudaDeviceIndex(device);
    if (index === null) {
      return null;
    }
    const options = deviceOptions ?? [];
    // With a single GPU there is no ambiguity to resolve, so we show nothing.
    if (options.length <= 1) {
      return null;
    }
    const name = device ? getDeviceNameLabels(options)[device] : undefined;
    if (!name) {
      return null;
    }
    return { index, name };
  }, [device, deviceOptions]);
};
