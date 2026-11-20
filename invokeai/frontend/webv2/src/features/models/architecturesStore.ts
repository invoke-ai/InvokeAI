/** Loads the architecture table on demand, per account; kept off the `index` barrel's Launchpad startup graph. */
export {
  ensureArchitectureCapabilitiesLoaded,
  getArchitectureCapabilitiesSnapshot,
  subscribeArchitectureCapabilities,
  useArchitectureCapabilitiesSelector,
  type ArchitectureCapabilitiesSnapshot,
} from './data/architectureCapabilitiesStore';
