import type { AutoSwitchMode } from '@features/queue/contracts';
import type { WorkbenchLanguage } from '@platform/i18n/languages';
import type { LogLevel, LogNamespace } from '@platform/logging/contracts';
import type { SettingsTarget } from '@platform/ui/settings/contracts';
import type { PromptFontSize } from '@theme/scale';
import type { WorkbenchThemeId } from '@theme/themes';
import type { ProjectSortId, ProjectsViewId } from '@workbench/launchpad/projects/projectLibraryView';

export const DEFAULT_SELECT_OBJECT_AUTO_UPDATE = true;

export type { ProjectSortId, ProjectsViewId } from '@workbench/launchpad/projects/projectLibraryView';

export type { WorkbenchLanguage } from '@platform/i18n/languages';

export type WorkflowLibrarySortId = 'updated' | 'opened' | 'created' | 'name' | 'lastRun';

/** The library order's direction; changing the key resets it to the key's natural one (dates newest first, names A to Z). */
export type WorkflowLibrarySortDirection = 'asc' | 'desc';

export type WorkflowLibraryCardSize = 'small' | 'large';

export type WorkflowLibraryLayout = 'grid' | 'list';

export type WorkflowLibraryThumbnailFit = 'cover' | 'contain';

export type StartupDestination = 'launchpad' | 'recentProject';

/** Exported with the project; personal choices belong in {@link WorkbenchPreferences} so imports cannot overwrite them. */
export interface ProjectSettings {
  /** Generation flags are pins: absent follows the user's default, then built-in (`getEffectiveProjectSettings`). */
  useCpuNoise?: boolean;
  useNsfwChecker?: boolean;
  useWatermark?: boolean;
  /** Projects saved before this existed read as `off`, so an upgrade never moves a settled selection. */
  viewerAutoSwitch: ViewerAutoSwitchMode;
}

export type ViewerAutoSwitchMode = AutoSwitchMode;

export type EffectiveProjectSettings = Required<ProjectSettings>;

/** User defaults per owner (widget type id, or `generate` for {@link ProjectSettings} flags); layerable keys only. */
export type ScopedDefaults = Readonly<Record<string, Readonly<Record<string, unknown>>>>;

/** Local copy of RebalancePreset keeps generation out of Launchpad's bundle; generation revalidates on read. */
export interface StoredRebalancePreset {
  id: string;
  label: string;
  weights: string;
  multiplier: number;
}

/** Kept local for the same bundle reason as {@link StoredRebalancePreset}. */
export interface StoredAspectRatioPreset {
  id: string;
  label: string;
  width: number;
  height: number;
}

/** `values` is opaque here for the same bundle reason as {@link StoredRebalancePreset}. */
export interface StoredGeneratePreset {
  id: string;
  label: string;
  values: Record<string, unknown>;
}

export interface WorkbenchPreferences {
  betaNoticeAcknowledged: boolean;
  whatsNewSeenVersion: string | null;
  themeId: WorkbenchThemeId;
  reduceMotion: boolean;
  showFocusRegionHighlight: boolean;
  confirmImageDeletion: boolean;
  galleryDestinationNoticeShown: boolean;
  /** Auto-switch the Invoke source/destination to match the surface being edited; locks always win. */
  autoSwitchInvocationRoute: boolean;
  queueJobsScope: 'active-project' | 'all';
  /** What opening the app's bare URL shows; deep links, reloads and later visits to Home are never redirected. */
  startupDestination: StartupDestination;
  language: WorkbenchLanguage;
  enableInformationalPopovers: boolean;
  enableModelDescriptions: boolean;
  notifyOnEnqueue: boolean;
  exportIncludesWorkspace: boolean;
  installInPlace: boolean;
  preferNumericAttentionStyle: boolean;
  showPromptSyntaxHighlighting: boolean;
  promptFontSize: PromptFontSize;
  developerLogEnabled: boolean;
  developerLogLevel: LogLevel;
  developerLogNamespaces: LogNamespace[];
  developerConsoleOutputEnabled: boolean;
  developerPerformanceTimingsEnabled: boolean;
  workflowGroupNodesByCategory: boolean;
  workflowSnapToGrid: boolean;
  workflowShowMinimap: boolean;
  workflowValidateConnections: boolean;
  workflowEdgeStyle: 'curved' | 'square';
  workflowEdgesBehindNodes: boolean;
  workflowLibrarySort: WorkflowLibrarySortId;
  workflowLibrarySortDirection: WorkflowLibrarySortDirection;
  workflowLibraryCardSize: WorkflowLibraryCardSize;
  workflowLibraryLayout: WorkflowLibraryLayout;
  workflowLibraryThumbnailFit: WorkflowLibraryThumbnailFit;
  highContrast: boolean;
  /** Renamed from `selectObjectAutoProcess`, whose stored `false` cannot tell a choice from the old default. */
  selectObjectAutoUpdate: boolean;
  customHotkeys: Record<string, string[]>;
  /** Generate panel section open/closed overrides keyed by section id; absent = section default. */
  generateSectionsOpen: Record<string, boolean>;
  launchpadProjectsView: ProjectsViewId;
  launchpadProjectsSort: ProjectSortId;
  /** Projects pinned to the top of the library, oldest pin first. */
  launchpadPinnedProjectIds: string[];
  aspectRatioPresets: StoredAspectRatioPreset[];
  krea2RebalancePresets: StoredRebalancePreset[];
  generatePresets: StoredGeneratePreset[];
  generatePresetsIncludePrompts: boolean;
  /** `${base}:${controlType}` → model key. */
  controlModelMemory: Record<string, string>;
  /** Canvas owns the shape and normalizes it on read (`canvas-operations/toolMemory.ts`). */
  canvasToolMemory: Readonly<Record<string, unknown>>;
  scopedDefaults: ScopedDefaults;
  antialiasProgressImages: boolean;
}

export type SettingsSectionId = string;
export interface SettingsDestination {
  sectionId: SettingsSectionId;
  entryId?: string;
  target?: SettingsTarget;
}
