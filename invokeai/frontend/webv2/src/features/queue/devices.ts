/** Keep device config outside queue/react so progress consumers do not load settings-only transport/store code. */
export {
  getDeviceLabel,
  getDeviceNameLabels,
  resolveRandDeviceMetadata,
  type DeviceLabel,
  type GenerationDeviceOption,
} from './core/deviceLabels';
export {
  type GenerationDevicesSetting,
  type GenerationDevicesSnapshot,
  getGenerationDevicesSnapshot,
  refreshGenerationDevices,
  updateGenerationDevices,
  useGenerationDevices,
} from './data/generationDevicesStore';
export { useDeviceLabel } from './ui/useDeviceLabel';
