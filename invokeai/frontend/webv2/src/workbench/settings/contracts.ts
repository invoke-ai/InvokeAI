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

/** User-tunable appearance + behavior preferences surfaced in the Settings modal. */
export interface WorkbenchPreferences {
  themeId: WorkbenchThemeId;
  reduceMotion: boolean;
  showFocusRegionHighlight: boolean;
  confirmImageDeletion: boolean;
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
}
