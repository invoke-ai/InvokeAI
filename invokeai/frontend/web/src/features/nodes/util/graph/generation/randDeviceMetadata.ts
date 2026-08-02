import type { RootState } from 'app/store/store';
import { appInfoApi } from 'services/api/endpoints/appInfo';

/**
 * The `rand_device` image-metadata value.
 *
 * This is descriptive metadata only — nothing dispatches on it. It used to be hardcoded to
 * `'cuda'` for any non-CPU noise, which is simply wrong on an Intel Arc machine. Derive the
 * accelerator from the backend's reported generation devices instead, falling back to `'cuda'`
 * when that data has not loaded yet so existing metadata is unchanged for Nvidia users.
 */
export const getRandDeviceMetadata = (state: RootState, shouldUseCpuNoise: boolean): string => {
  if (shouldUseCpuNoise) {
    return 'cpu';
  }
  const devices = appInfoApi.endpoints.getGenerationDeviceOptions.select()(state).data;
  const deviceType = devices?.[0]?.device.split(':')[0];
  return deviceType && deviceType !== 'cpu' ? deviceType : 'cuda';
};
