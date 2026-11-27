/**
 * Pure architecture facts for pure cores: base labels, the served architecture table and the VAE pairing question.
 * No data, transport or React. The table loads through `architecturesStore`.
 */
export {
  architectureAcceptsVae,
  getArchitectureCapabilityRow,
  getArchitectureFeatures,
  hasArchitectureCapabilities,
  withArchitectureCapabilities,
  type ArchitectureCapabilitiesRow,
} from './core/architectureCapabilities';
export { getModelBaseLabel, isFluxFillModel } from './core/baseIdentity';
