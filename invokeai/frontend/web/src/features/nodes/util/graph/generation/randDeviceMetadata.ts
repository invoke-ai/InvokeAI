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
  // Only answer when every generation device is the same accelerator. Under a mixed
  // `generation_devices` list the client cannot know which one the session landed on, so
  // guessing from the first entry would be no more truthful than the old hardcoded value.
  const devices = appInfoApi.endpoints.getGenerationDeviceOptions.select()(state).data;
  const types = new Set(devices?.map(({ device }) => device.split(':')[0]).filter((type) => type !== 'cpu'));
  const [deviceType] = types;
  return types.size === 1 && deviceType ? deviceType : 'cuda';
};
