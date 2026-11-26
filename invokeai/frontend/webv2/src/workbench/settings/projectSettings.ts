/** Pure, so the codec, reducers and settings store share it; it boots on both routes. */
import type { EffectiveProjectSettings, ProjectSettings, ViewerAutoSwitchMode } from '@workbench/settings/contracts';

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

/** Its defaults layer over `ProjectSettings`, never over the Generate widget's own values. */
export const PROJECT_SETTINGS_OWNER = 'generate';

export interface LayeredSettingKey {
  builtIn: unknown;
  read: (value: unknown) => unknown;
}

export interface LayeredSettingsTable {
  widgetValues: Readonly<Record<string, Readonly<Record<string, LayeredSettingKey>>>>;
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
 * Frozen as of `settingsLayering: 1`: changing an entry would change what old documents lose; add new keys instead.
 * Built on first use, since a module-scope table of calls would make this boot module side-effectful.
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
        scrollInput: oneOf('auto', ['mouse', 'trackpad']),
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

export const getLayeredKeys = (ownerId: string): Readonly<Record<string, LayeredSettingKey>> | undefined => {
  const table = getLayeredSettings();
  return ownerId === PROJECT_SETTINGS_OWNER ? table.projectSettings : table.widgetValues[ownerId];
};

/** A record, not the policy's list, to keep the Queue contracts out of the boot graph; exhaustive by type. */
const VIEWER_AUTO_SWITCH_MODES: Readonly<Record<ViewerAutoSwitchMode, true>> = {
  latest: true,
  off: true,
  progress: true,
};

export const isViewerAutoSwitchMode = (value: unknown): value is ViewerAutoSwitchMode =>
  typeof value === 'string' && Object.hasOwn(VIEWER_AUTO_SWITCH_MODES, value);

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

/** Memoized on the settings object, so unchanged reads return the same object. */
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
