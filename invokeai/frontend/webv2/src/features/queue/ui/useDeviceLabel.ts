import type { DeviceLabel } from '@features/queue/core/deviceLabels';

import { getDeviceLabel } from '@features/queue/core/deviceLabels';
import { useGenerationDeviceOptions } from '@features/queue/data/generationDevicesStore';
import { useMemo } from 'react';

/**
 * Resolve a queue item's or progress event's device to a badge index + full name.
 *
 * Returns null whenever there is nothing worth showing — most importantly on a
 * single-GPU install, where every item runs on the same GPU and a label would be
 * pure noise. Callers can therefore render unconditionally on a non-null result.
 */
export const useDeviceLabel = (device: string | null | undefined): DeviceLabel | null => {
  const options = useGenerationDeviceOptions();

  return useMemo(() => getDeviceLabel(device, options), [device, options]);
};
