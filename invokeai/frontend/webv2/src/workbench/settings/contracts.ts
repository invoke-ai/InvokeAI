import type { WorkbenchLanguage } from '@platform/i18n/languages';
import type { LogLevel, LogNamespace } from '@platform/logging/contracts';
import type { SettingsTarget } from '@platform/ui/settings/contracts';
import type { PromptFontSize } from '@theme/scale';
import type { WorkbenchThemeId } from '@theme/themes';
import type { ProjectSortId, ProjectsViewId } from '@workbench/launchpad/projects/projectLibraryView';

export type { ProjectSortId, ProjectsViewId } from '@workbench/launchpad/projects/projectLibraryView';

export type { WorkbenchLanguage } from '@platform/i18n/languages';

/** What the workflow library's own templates are ordered by (dates newest first, names A to Z). */
export type WorkflowLibrarySortId = 'updated' | 'opened' | 'created' | 'name';

/** How workflow thumbnails fill their 3:2 box: cropped to fill it, or letterboxed to fit whole. */
export type WorkflowLibraryThumbnailFit = 'cover' | 'contain';

/** What a launch of the app's bare URL opens: the Launchpad, or the most recent project. */
export type StartupDestination = 'launchpad' | 'recentProject';

/**
 * Export document generation settings with the project; personal editor choices belong in {@link
 * WorkbenchPreferences} so imports cannot overwrite them.
 */
export interface ProjectSettings {
  /**
   * The three generation flags are this project's pins: absent follows the user's default
   * (`WorkbenchPreferences.scopedDefaults.generate`), then the built-in. Read them through
   * `getEffectiveProjectSettings`.
   */
  useCpuNoise?: boolean;
  /** Blur images the NSFW checker flags before they are saved. */
  useNsfwChecker?: boolean;
  /** Stamp an invisible watermark into every saved image. */
  useWatermark?: boolean;
  /**
   * What the Preview switches to while work runs; the Preview owns this control, so it is not in Settings. Projects
   * saved before it existed read as `off`, so an upgrade never starts moving a selection the user had settled on.
   */
  viewerAutoSwitch: ViewerAutoSwitchMode;
}

/**
 * `latest`: each finished result takes the selection. `progress`: the same, and live frames show while it runs.
 * `off`: results land in the Gallery only (one still takes an empty selection). The canvas staging bar's values.
 */
export type ViewerAutoSwitchMode = 'off' | 'latest' | 'progress';

/** {@link ProjectSettings} with every layered flag resolved: project pin, then the user's default, then built-in. */
export type EffectiveProjectSettings = Required<ProjectSettings>;

/**
 * The user's defaults for per-project settings, keyed by owner: a widget type id (`gallery`, `canvas`, `preview`,
 * `image-map`), and `generate` for the layered {@link ProjectSettings} flags the Generate widget's settings show (they
 * layer over the project settings, never over the Generate widget's values). Each record holds only layerable keys
 * the user chose (`getLayeredSettings`); owners normalize the values on read, as they do their widget values.
 */
export type ScopedDefaults = Readonly<Record<string, Readonly<Record<string, unknown>>>>;

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
 * A user-named aspect ratio as it is persisted, kept local for the same bundle reason as {@link StoredRebalancePreset}.
 * Generation owns the ratio rules and revalidates the terms on read.
 */
export interface StoredAspectRatioPreset {
  id: string;
  label: string;
  width: number;
  height: number;
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
  /** The note that browsing a board no longer changes where new results go was shown; once per account. */
  galleryDestinationNoticeShown: boolean;
  /** Auto-switch the Invoke source/destination to match the surface being edited; locks always win. */
  autoSwitchInvocationRoute: boolean;
  queueJobsScope: 'active-project' | 'all';
  /** What opening the app's bare URL shows; deep links, reloads and later visits to Home are never redirected. */
  startupDestination: StartupDestination;
  language: WorkbenchLanguage;
  enableInformationalPopovers: boolean;
  enableModelDescriptions: boolean;
  /** Toast when an invocation is added to the queue. The notification center records it regardless. */
  notifyOnEnqueue: boolean;
  /** Project export carries the exporter's workspace (layout, generation settings, prompt history); the last choice. */
  exportIncludesWorkspace: boolean;
  /** Add Models installs local files where they are instead of moving them into the models folder; the last choice. */
  installInPlace: boolean;
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
  /** Order of the workflow library's own templates. */
  workflowLibrarySort: WorkflowLibrarySortId;
  /** How workflow library and project cards show their thumbnails. */
  workflowLibraryThumbnailFit: WorkflowLibraryThumbnailFit;
  /** Raise text and border contrast on every theme (a11y). */
  highContrast: boolean;
  /** Select Object processes each prompt edit automatically; kept across sessions, Reset, and reloads. */
  selectObjectAutoProcess: boolean;
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
  /** User-named aspect ratios (ratios only, never sizes), in the order they were saved. */
  aspectRatioPresets: StoredAspectRatioPreset[];
  /** User-saved Krea-2 conditioning rebalance curves, in the order they were saved. */
  krea2RebalancePresets: StoredRebalancePreset[];
  /** User-saved Generate settings snapshots ("recipes"), in the order they were saved. */
  generatePresets: StoredGeneratePreset[];
  /** Whether a new Generate preset keeps the prompts too: the save dialog's last choice. Not shown in Settings. */
  generatePresetsIncludePrompts: boolean;
  /**
   * The last control model picked per main-model base and control type (`${base}:${controlType}` → model key);
   * control layers left on automatic use it. Not shown in Settings.
   */
  controlModelMemory: Record<string, string>;
  /**
   * Canvas brush, eraser, shape and text options, restored whenever a canvas engine starts. Canvas owns its shape and
   * normalizes it on every read (`canvas-operations/toolMemory.ts`); here it is only a plain object. Not shown in
   * Settings.
   */
  canvasToolMemory: Readonly<Record<string, unknown>>;
  /**
   * Per-project settings' user defaults (see {@link ScopedDefaults}): a project without its own value for a setting
   * follows these, then the built-in. Not shown as such in Settings.
   */
  scopedDefaults: ScopedDefaults;
  /** Smooth the low-resolution live frames shown while an image generates. */
  antialiasProgressImages: boolean;
}

export type SettingsSectionId = string;
export interface SettingsDestination {
  sectionId: SettingsSectionId;
  entryId?: string;
  target?: SettingsTarget;
}
