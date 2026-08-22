/**
 * Generation-device reads, kept out of the broad `queue/react` barrel.
 *
 * The device options come from a server app-config endpoint that only the settings
 * surface and the device labels need. Exporting them from `react` pulled the store into
 * the activation payload of every widget that imports that barrel (the gallery adapter
 * imports it for progress targets), which the architecture performance gate flags.
 */
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
