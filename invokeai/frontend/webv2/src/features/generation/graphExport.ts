/** Offline graph export for tooling such as `pnpm graphs:export`; the browser bundle never loads it. */
export {
  compileGraphExportCase,
  withArchitectureCapabilities,
  type CompiledGraphExportCase,
  type GraphExportCase,
  type GraphExportEnvironment,
  type GraphExportMode,
} from './core/graphExport';
export type { ArchitectureCapabilitiesRow } from './core/architectureCapabilities';
