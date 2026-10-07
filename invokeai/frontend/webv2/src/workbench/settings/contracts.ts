import type { WorkbenchLanguage } from '@platform/i18n/languages';
import type { LogLevel, LogNamespace } from '@platform/logging/contracts';
import type { SettingsTarget } from '@platform/ui/settings/contracts';
import type { PromptFontSize } from '@theme/scale';
import type { WorkbenchThemeId } from '@theme/themes';
import type { ProjectSortId, ProjectsViewId } from '@workbench/launchpad/projects/projectLibraryView';

export type { ProjectSortId, ProjectsViewId } from '@workbench/launchpad/projects/projectLibraryView';

export type { WorkbenchLanguage } from '@platform/i18n/languages';

/**
 * Export document generation settings with the project; personal editor choices belong in {@link
 * WorkbenchPreferences} so imports cannot overwrite them.
 */
export interface ProjectSettings {
  useCpuNoise: boolean;
  antialiasProgressImages: boolean;
  /** Preview owns this live toggle; persist it per project rather than exposing it in Settings. */
  showProgressImagesInViewer: boolean;
}

/**
 * Store the RebalancePreset shape locally to keep generation out of Launchpad's bundle. Generation owns and
 * revalidates weights on read.
 */
export interface StoredRebalancePreset {
  id: string;
  label: string;
  weights: string;
  multiplier: number;
}

/**
 * A named Generate settings snapshot ("recipe") as it is persisted. `values` is
 * opaque to Settings for the same bundle reason as {@link StoredRebalancePreset}:
 * the generation feature owns the shape and re-normalizes it on apply.
 */
export interface StoredGeneratePreset {
  id: string;
  label: string;
  values: Record<string, unknown>;
}

/** User-tunable appearance + behavior preferences surfaced in the Settings modal. */
export interface WorkbenchPreferences {
  /** The alpha-build notice was dismissed; shown once per account until then. */
  alphaNoticeAcknowledged: boolean;
  /** Server version whose What's New notes were last dismissed; a different version shows them once. */
  whatsNewSeenVersion: string | null;
  themeId: WorkbenchThemeId;
  reduceMotion: boolean;
  showFocusRegionHighlight: boolean;
  confirmImageDeletion: boolean;
  /** Starred images and videos are skipped by every delete, and a board delete moves them to Uncategorized. */
  protectStarredMedia: boolean;
  /** Auto-switch the Invoke source/destination to match the surface being edited; locks always win. */
  autoSwitchInvocationRoute: boolean;
  queueJobsScope: 'active-project' | 'all';
  language: WorkbenchLanguage;
  enableInformationalPopovers: boolean;
  enableModelDescriptions: boolean;
  /** Toast when an invocation is added to the queue. The notification center records it regardless. */
  notifyOnEnqueue: boolean;
  /** Write numeric attention weights (`(word)1.1`) when the attention hotkeys insert them. */
  preferNumericAttentionStyle: boolean;
  /** Color prompt syntax in prompt fields; changes rendering only. */
  showPromptSyntaxHighlighting: boolean;
  /** Text size of the prompt fields. */
  promptFontSize: PromptFontSize;
  developerLogEnabled: boolean;
  developerLogLevel: LogLevel;
  developerLogNamespaces: LogNamespace[];
  /** Mirror recorded entries to the browser console; obeys the same recording filters. */
  developerConsoleOutputEnabled: boolean;
  developerPerformanceTimingsEnabled: boolean;
  /** Group add-node search results under their categories; off lists them flat by relevance. */
  workflowGroupNodesByCategory: boolean;
  /** Always snap workflow nodes to the grid (Ctrl snaps temporarily when off). */
  workflowSnapToGrid: boolean;
  /** Show the minimap in the workflow editor. */
  workflowShowMinimap: boolean;
  /** Reject workflow connections with incompatible field types. */
  workflowValidateConnections: boolean;
  /** Connection line rendering in the workflow editor. */
  workflowEdgeStyle: 'curved' | 'square';
  /** Keep connections of a selected node beneath nodes instead of raising them over node controls. */
  workflowEdgesBehindNodes: boolean;
  /** Raise text and border contrast on every theme (a11y). */
  highContrast: boolean;
  /** Account-bound overrides keyed by hotkey id (`app.invoke`, `gallery.galleryNavLeft`, etc.). */
  customHotkeys: Record<string, string[]>;
  /** Generate panel section open/closed overrides keyed by section id; absent = section default. */
  generateSectionsOpen: Record<string, boolean>;
  /** How the Launchpad's project library is laid out. */
  launchpadProjectsView: ProjectsViewId;
  /** What the Launchpad's project library is ordered and bucketed by. */
  launchpadProjectsSort: ProjectSortId;
  /** Projects pinned to the top of the library, oldest pin first. */
  launchpadPinnedProjectIds: string[];
  /** User-saved Krea-2 conditioning rebalance curves, in the order they were saved. */
  krea2RebalancePresets: StoredRebalancePreset[];
  /** User-saved Generate settings snapshots ("recipes"), in the order they were saved. */
  generatePresets: StoredGeneratePreset[];
}

export type SettingsSectionId = string;
export interface SettingsDestination {
  sectionId: SettingsSectionId;
  entryId?: string;
  target?: SettingsTarget;
}
