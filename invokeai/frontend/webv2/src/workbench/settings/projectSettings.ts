/**
 * Per-project settings layered over the user's defaults: which keys layer, the project settings as stored (the
 * project's pins) and as read (resolved over the user's defaults). Pure, so the document codec, reducers and the
 * settings store share it without loading each other; it boots on both routes, so the table adds no startup module.
 */
import type { EffectiveProjectSettings, ProjectSettings, ViewerAutoSwitchMode } from '@workbench/settings/contracts';

/** What each project setting is when neither the project nor the user's default says otherwise. */
export const BUILT_IN_PROJECT_SETTINGS: EffectiveProjectSettings = {
  useCpuNoise: true,
  useNsfwChecker: false,
  useWatermark: false,
  // New projects; a stored project without the field reads as `off` (see normalizeProjectSettings).
  viewerAutoSwitch: 'progress',
};

/** The project settings that layer over the user's `scopedDefaults.generate`; `viewerAutoSwitch` is project-only. */
export const LAYERED_PROJECT_SETTING_KEYS = ['useCpuNoise', 'useNsfwChecker', 'useWatermark'] as const;
export type LayeredProjectSettingKey = (typeof LAYERED_PROJECT_SETTING_KEYS)[number];

/**
 * The `scopedDefaults` owner of the layered project settings: the Generate widget's type id, whose settings show them.
 * Its defaults layer over `ProjectSettings`, never over the Generate widget's own values.
 */
export const PROJECT_SETTINGS_OWNER = 'generate';

/** One layerable key as of `settingsLayering: 1`: its built-in, and how its owner read a stored value then. */
export interface LayeredSettingKey {
  builtIn: unknown;
  read: (value: unknown) => unknown;
}

export interface LayeredSettingsTable {
  /** Per widget type id, keys of the instance's `state.values`. */
  widgetValues: Readonly<Record<string, Readonly<Record<string, LayeredSettingKey>>>>;
  /** Keys of `ProjectSettings`, owned by {@link PROJECT_SETTINGS_OWNER}. */
  projectSettings: Readonly<Record<LayeredProjectSettingKey, LayeredSettingKey>>;
}

const typedBoolean = (builtIn: boolean): LayeredSettingKey => ({
  builtIn,
  read: (value) => (typeof value === 'boolean' ? value : builtIn),
});
const onlyTrue = (): LayeredSettingKey => ({ builtIn: false, read: (value) => value === true });
const unlessFalse = (): LayeredSettingKey => ({ builtIn: true, read: (value) => value !== false });
const oneOf = (builtIn: string, others: readonly string[]): LayeredSettingKey => ({
  builtIn,
  read: (value) => (typeof value === 'string' && others.includes(value) ? value : builtIn),
});

let layeredSettingsV1: LayeredSettingsTable | undefined;

/**
 * Every layerable key, per owner, as of `settingsLayering: 1`. The one list of what layers: resolution merges only
 * these keys from the user's defaults, the settings store keeps only these, and the document codec drops a stored
 * value the owner read as the built-in from a document without the marker. The codec's meaning is frozen: a later
 * change to an owner's built-in or normalizer does not edit an entry here (it would change what old documents lose);
 * a new layerable key is added, and tests keep the keys equal to the settings contributions'. Built on first use: a
 * module-scope table of calls would make this boot module side-effectful to the bundler.
 */
export const getLayeredSettings = (): LayeredSettingsTable =>
  (layeredSettingsV1 ??= {
    projectSettings: {
      useCpuNoise: typedBoolean(true),
      useNsfwChecker: typedBoolean(false),
      useWatermark: typedBoolean(false),
    },
    widgetValues: {
      canvas: {
        bboxOverlay: typedBoolean(false),
        clipToBbox: typedBoolean(false),
        invertBrushSizeScroll: typedBoolean(false),
        outputOnlyMaskedRegions: typedBoolean(true),
        ruleOfThirds: typedBoolean(false),
        showBbox: typedBoolean(true),
        showCheckerboard: typedBoolean(true),
        showGrid: typedBoolean(false),
        showProgressOnCanvas: typedBoolean(true),
        snapToGrid: typedBoolean(true),
      },
      gallery: {
        boardOrderBy: oneOf('created_at', ['board_name']),
        boardOrderDir: oneOf('DESC', ['ASC']),
        followSelectedBoard: onlyTrue(),
        // The size slider shows 100 minus this.
        imageDensityPercent: {
          builtIn: 50,
          read: (value) =>
            typeof value === 'number' && Number.isFinite(value) ? Math.min(100, Math.max(0, value)) : 50,
        },
        imageOrderDir: oneOf('DESC', ['ASC']),
        middleClickOpensInNewTab: onlyTrue(),
        paginationMode: oneOf('infinite', ['paginated']),
        showArchivedBoards: typedBoolean(false),
        showDateBoards: typedBoolean(false),
        showImageDimensions: typedBoolean(false),
        showOtherProjectBoards: typedBoolean(false),
        showPendingItems: typedBoolean(true),
        thumbnailFit: oneOf('square', ['aspect']),
      },
      'image-map': {
        clickSelectsCluster: onlyTrue(),
        // Null hands the choice to the server's heuristic.
        clusterEps: {
          builtIn: null,
          read: (value) => (typeof value === 'number' && value >= 0.01 && value <= 2 ? value : null),
        },
        showClusterLabels: unlessFalse(),
      },
      preview: {
        comparisonMode: oneOf('slider', ['side-by-side', 'hover']),
        filmstripVisible: unlessFalse(),
      },
    },
  });

/** The layerable keys of an owner's `scopedDefaults` record: a widget type's values, or the project settings. */
export const getLayeredKeys = (ownerId: string): Readonly<Record<string, LayeredSettingKey>> | undefined => {
  const table = getLayeredSettings();
  return ownerId === PROJECT_SETTINGS_OWNER ? table.projectSettings : table.widgetValues[ownerId];
};

export const isViewerAutoSwitchMode = (value: unknown): value is ViewerAutoSwitchMode =>
  value === 'off' || value === 'latest' || value === 'progress';

/** Keeps the project's own pins (absent follows the user's default) and drops everything else it does not own. */
export const normalizeProjectSettings = (settings?: Partial<ProjectSettings>): ProjectSettings => {
  const normalized: ProjectSettings = {
    // Absent means a project saved before the mode existed, whatever its old progress toggle said: it starts `off`.
    viewerAutoSwitch: isViewerAutoSwitchMode(settings?.viewerAutoSwitch) ? settings.viewerAutoSwitch : 'off',
  };
  for (const key of LAYERED_PROJECT_SETTING_KEYS) {
    const value = settings?.[key];
    if (typeof value === 'boolean') {
      normalized[key] = value;
    }
  }
  return normalized;
};

// Created on first use: a module-scope `new WeakMap()` reads a global, which makes the module side-effectful to
// Rolldown and keeps it (and every barrel re-exporting through it) out of tree-shaking.
let effectiveProjectSettingsCache:
  | WeakMap<
      ProjectSettings,
      { defaults: Readonly<Record<string, unknown>> | undefined; effective: EffectiveProjectSettings }
    >
  | undefined;

/**
 * Resolves each layered flag as project pin, then the user's default, then built-in. Memoized on the settings
 * object, so repeated reads of an unchanged project and defaults return the same object.
 */
export const getEffectiveProjectSettings = (
  settings: ProjectSettings,
  defaults: Readonly<Record<string, unknown>> | undefined
): EffectiveProjectSettings => {
  effectiveProjectSettingsCache ??= new WeakMap();
  const cached = effectiveProjectSettingsCache.get(settings);
  if (cached && cached.defaults === defaults) {
    return cached.effective;
  }
  const effective: EffectiveProjectSettings = {
    ...BUILT_IN_PROJECT_SETTINGS,
    viewerAutoSwitch: settings.viewerAutoSwitch,
  };
  for (const key of LAYERED_PROJECT_SETTING_KEYS) {
    const pinned = settings[key];
    const fallback = defaults?.[key];
    effective[key] = typeof pinned === 'boolean' ? pinned : typeof fallback === 'boolean' ? fallback : effective[key];
  }
  effectiveProjectSettingsCache.set(settings, { defaults, effective });
  return effective;
};
