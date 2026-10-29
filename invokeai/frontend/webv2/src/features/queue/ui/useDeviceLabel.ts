import type { DeviceLabel } from '@features/queue/core/deviceLabels';

import { getDeviceLabel } from '@features/queue/core/deviceLabels';
import { useGenerationDeviceOptions } from '@features/queue/data/generationDevicesStore';
import { useMemo } from 'react';

/** Return device badge details only when multiple reported accelerators make identification useful. */
export const useDeviceLabel = (device: string | null | undefined): DeviceLabel | null => {
  const options = useGenerationDeviceOptions();

  return useMemo(() => getDeviceLabel(device, options), [device, options]);
};
