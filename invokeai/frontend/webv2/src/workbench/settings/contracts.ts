import type { WorkbenchLanguage } from '@platform/i18n/languages';
import type { WorkbenchThemeId } from '@theme/themes';
import type { DeveloperLogLevel, DeveloperLogNamespace } from '@workbench/diagnostics/contracts';

export type { WorkbenchLanguage } from '@platform/i18n/languages';

export interface ProjectSettings {
  useCpuNoise: boolean;
  showProgressDetails: boolean;
  antialiasProgressImages: boolean;
  showProgressImagesInViewer: boolean;
  preferNumericAttentionStyle: boolean;
  showPromptSyntaxHighlighting: boolean;
}

/**
 * A saved conditioning-rebalance curve as it is persisted.
 *
 * Structurally the feature's `RebalancePreset`, declared here rather than imported:
 * these preferences load on the launchpad route, and a value import from
 * `@features/generation` would pull that feature's whole core into the initial bundle.
 * Settings stores the record; the generation feature owns what `weights` means and
 * re-validates it on read.
 */
export interface StoredRebalancePreset {
  id: string;
  label: string;
  weights: string;
  multiplier: number;
}

/** User-tunable appearance + behavior preferences surfaced in the Settings modal. */
export interface WorkbenchPreferences {
  themeId: WorkbenchThemeId;
  reduceMotion: boolean;
  showFocusRegionHighlight: boolean;
  confirmImageDeletion: boolean;
  /** Auto-switch the Invoke source/destination to match the surface being edited; locks always win. */
  autoSwitchInvocationRoute: boolean;
  queueJobsScope: 'active-project' | 'all';
  language: WorkbenchLanguage;
  enableInformationalPopovers: boolean;
  enableModelDescriptions: boolean;
  developerLogEnabled: boolean;
  developerLogLevel: DeveloperLogLevel;
  developerLogNamespaces: DeveloperLogNamespace[];
  developerPerformanceTimingsEnabled: boolean;
  /** Always snap workflow nodes to the grid (Ctrl snaps temporarily when off). */
  workflowSnapToGrid: boolean;
  /** Show the minimap in the workflow editor. */
  workflowShowMinimap: boolean;
  /** Reject workflow connections with incompatible field types. */
  workflowValidateConnections: boolean;
  /** Connection line rendering in the workflow editor. */
  workflowEdgeStyle: 'curved' | 'square';
  /** Account-bound overrides keyed by hotkey id (`app.invoke`, `gallery.galleryNavLeft`, etc.). */
  customHotkeys: Record<string, string[]>;
  /** Generate panel section open/closed overrides keyed by section id; absent = section default. */
  generateSectionsOpen: Record<string, boolean>;
  /** User-saved Krea-2 conditioning rebalance curves, in the order they were saved. */
  krea2RebalancePresets: StoredRebalancePreset[];
}
