import type { PayloadAction, Selector, UnknownAction } from '@reduxjs/toolkit';
import { createSelector, createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { isPlainObject } from 'es-toolkit';
import type { RgbaColor } from 'features/controlLayers/store/types';
import { RGBA_BLACK, RGBA_WHITE, zRgbaColor } from 'features/controlLayers/store/types';
import { assert } from 'tsafe';
import { z } from 'zod';

import {
  canvasCreated,
  canvasMultiCanvasMigrated,
  canvasRemoved,
  MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER,
} from './canvasSlice';

const zAutoSwitchMode = z.enum(['off', 'switch_on_start', 'switch_on_finish']);
export type AutoSwitchMode = z.infer<typeof zAutoSwitchMode>;

const zCanvasSharedSettings = z.object({
  /**
   * Whether to show HUD (Heads-Up Display) on the canvas.
   */
  showHUD: z.boolean(),
  /**
   * Whether to clip lines and shapes to the generation bounding box. If disabled, lines and shapes will be clipped to
   * the canvas bounds.
   */
  clipToBbox: z.boolean(),
  /**
   * Whether to show a dynamic grid on the canvas. If disabled, a checkerboard pattern will be shown instead.
   */
  dynamicGrid: z.boolean(),
  /**
   * Whether to invert the scroll direction when adjusting the brush or eraser width with the scroll wheel.
   */
  invertScrollForToolWidth: z.boolean(),
  /**
   * Whether to composite inpainted/outpainted regions back onto the source image when saving canvas generations.
   *
   * If disabled, inpainted/outpainted regions will be saved with a transparent background.
   *
   * When `sendToCanvas` is disabled, this setting is ignored, masked regions will always be composited.
   */
  outputOnlyMaskedRegions: z.boolean(),
  /**
   * Whether to automatically process the operations like filtering and auto-masking.
   */
  autoProcess: z.boolean(),
  /**
   * The snap-to-grid setting for the canvas.
   */
  snapToGrid: z.boolean(),
  /**
   * Whether to show progress on the canvas when generating images.
   */
  showProgressOnCanvas: z.boolean(),
  /**
   * Whether to show the bounding box overlay on the canvas.
   */
  bboxOverlay: z.boolean(),
  /**
   * Whether to preserve the masked region instead of inpainting it.
   */
  preserveMask: z.boolean(),
  /**
   * Whether to show only raster layers while staging.
   */
  isolatedStagingPreview: z.boolean(),
  /**
   * Whether to show only the selected layer while filtering, transforming, or doing other operations.
   */
  isolatedLayerPreview: z.boolean(),
  /**
   * Whether to use pressure sensitivity for the brush and eraser tool when a pen device is used.
   */
  pressureSensitivity: z.boolean(),
  /**
   * Whether to show the rule of thirds composition guide overlay on the canvas.
   */
  ruleOfThirds: z.boolean(),
  /**
   * Whether to save all staging images to the gallery instead of keeping them as intermediate images.
   */
  saveAllImagesToGallery: z.boolean(),
  /**
   * The auto-switch mode for the canvas staging area.
   */
  stagingAreaAutoSwitch: zAutoSwitchMode,
});
type CanvasSharedSettings = z.infer<typeof zCanvasSharedSettings>;

const zCanvasInstanceSettings = z.object({
  canvasId: z.string(),
  /**
   * The width of the brush tool.
   */
  brushWidth: z.int().gt(0),
  /**
   * The width of the eraser tool.
   */
  eraserWidth: z.int().gt(0),
  /**
   * The colors to use when drawing lines or filling shapes.
   */
  activeColor: z.enum(['bgColor', 'fgColor']),
  bgColor: zRgbaColor,
  fgColor: zRgbaColor,
});
type CanvasInstanceSettings = z.infer<typeof zCanvasInstanceSettings>;

const zCanvasSettingsState = z.object({
  _version: z.literal(1),
  shared: zCanvasSharedSettings,
  canvases: z.record(z.string(), zCanvasInstanceSettings),
});
type CanvasSettingsState = z.infer<typeof zCanvasSettingsState>;

const getInitialCanvasSharedSettings = (): CanvasSharedSettings => ({
  showHUD: true,
  clipToBbox: false,
  dynamicGrid: false,
  invertScrollForToolWidth: false,
  outputOnlyMaskedRegions: true,
  autoProcess: true,
  snapToGrid: true,
  showProgressOnCanvas: true,
  bboxOverlay: false,
  preserveMask: false,
  isolatedStagingPreview: true,
  isolatedLayerPreview: true,
  pressureSensitivity: true,
  ruleOfThirds: false,
  saveAllImagesToGallery: false,
  stagingAreaAutoSwitch: 'switch_on_start',
});
const getInitialCanvasInstanceSettings = (canvasId: string): CanvasInstanceSettings => ({
  canvasId,
  brushWidth: 50,
  eraserWidth: 50,
  activeColor: 'fgColor',
  bgColor: RGBA_BLACK,
  fgColor: RGBA_WHITE,
});
const getInitialCanvasSettingsState = (): CanvasSettingsState => ({
  _version: 1,
  shared: getInitialCanvasSharedSettings(),
  canvases: {},
});

type PayloadWithCanvasId<P> = P & { canvasId: string };
type CanvasPayloadAction<P> = PayloadAction<PayloadWithCanvasId<P>>;

const canvasSettingsSlice = createSlice({
  name: 'canvasSettings',
  initialState: getInitialCanvasSettingsState(),
  reducers: {
    settingsClipToBboxChanged: (state, action: PayloadAction<{ clipToBbox: boolean }>) => {
      const { clipToBbox } = action.payload;

      state.shared.clipToBbox = clipToBbox;
    },
    settingsDynamicGridToggled: (state) => {
      state.shared.dynamicGrid = !state.shared.dynamicGrid;
    },
    settingsShowHUDToggled: (state) => {
      state.shared.showHUD = !state.shared.showHUD;
    },
    settingsInvertScrollForToolWidthChanged: (state, action: PayloadAction<{ invertScrollForToolWidth: boolean }>) => {
      const { invertScrollForToolWidth } = action.payload;

      state.shared.invertScrollForToolWidth = invertScrollForToolWidth;
    },
    settingsOutputOnlyMaskedRegionsToggled: (state) => {
      state.shared.outputOnlyMaskedRegions = !state.shared.outputOnlyMaskedRegions;
    },
    settingsAutoProcessToggled: (state) => {
      state.shared.autoProcess = !state.shared.autoProcess;
    },
    settingsSnapToGridToggled: (state) => {
      state.shared.snapToGrid = !state.shared.snapToGrid;
    },
    settingsShowProgressOnCanvasToggled: (state) => {
      state.shared.showProgressOnCanvas = !state.shared.showProgressOnCanvas;
    },
    settingsBboxOverlayToggled: (state) => {
      state.shared.bboxOverlay = !state.shared.bboxOverlay;
    },
    settingsPreserveMaskToggled: (state) => {
      state.shared.preserveMask = !state.shared.preserveMask;
    },
    settingsIsolatedStagingPreviewToggled: (state) => {
      state.shared.isolatedStagingPreview = !state.shared.isolatedStagingPreview;
    },
    settingsIsolatedLayerPreviewToggled: (state) => {
      state.shared.isolatedLayerPreview = !state.shared.isolatedLayerPreview;
    },
    settingsPressureSensitivityToggled: (state) => {
      state.shared.pressureSensitivity = !state.shared.pressureSensitivity;
    },
    settingsRuleOfThirdsToggled: (state) => {
      state.shared.ruleOfThirds = !state.shared.ruleOfThirds;
    },
    settingsSaveAllImagesToGalleryToggled: (state) => {
      state.shared.saveAllImagesToGallery = !state.shared.saveAllImagesToGallery;
    },
    settingsStagingAreaAutoSwitchChanged: (
      state,
      action: PayloadAction<{ stagingAreaAutoSwitch: CanvasSharedSettings['stagingAreaAutoSwitch'] }>
    ) => {
      const { stagingAreaAutoSwitch } = action.payload;

      state.shared.stagingAreaAutoSwitch = stagingAreaAutoSwitch;
    },
  },
  extraReducers(builder) {
    builder.addCase(canvasCreated, (state, action) => {
      const canvasSettings = getInitialCanvasInstanceSettings(action.payload.canvasId);
      state.canvases[canvasSettings.canvasId] = canvasSettings;
    });
    builder.addCase(canvasRemoved, (state, action) => {
      delete state.canvases[action.payload.canvasId];
    });
    builder.addCase(canvasMultiCanvasMigrated, (state, action) => {
      const settings = state.canvases[MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER];
      if (!settings) {
        return;
      }
      settings.canvasId = action.payload.canvasId;
      state.canvases[settings.canvasId] = settings;
      delete state.canvases[MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER];
    });
  },
});

const canvasInstanceSettingsSlice = createSlice({
  name: 'canvasSettings',
  initialState: {} as CanvasInstanceSettings,
  reducers: {
    settingsBrushWidthChanged: (state, action: CanvasPayloadAction<{ brushWidth: number }>) => {
      const { brushWidth } = action.payload;

      state.brushWidth = Math.round(brushWidth);
    },
    settingsEraserWidthChanged: (state, action: CanvasPayloadAction<{ eraserWidth: number }>) => {
      const { eraserWidth } = action.payload;

      state.eraserWidth = Math.round(eraserWidth);
    },
    settingsActiveColorToggled: (state, _action: CanvasPayloadAction<unknown>) => {
      state.activeColor = state.activeColor === 'bgColor' ? 'fgColor' : 'bgColor';
    },
    settingsBgColorChanged: (state, action: CanvasPayloadAction<{ bgColor: Partial<RgbaColor> }>) => {
      const { bgColor } = action.payload;

      state.bgColor = { ...state.bgColor, ...bgColor };
    },
    settingsFgColorChanged: (state, action: CanvasPayloadAction<{ fgColor: Partial<RgbaColor> }>) => {
      const { fgColor } = action.payload;

      state.fgColor = { ...state.fgColor, ...fgColor };
    },
    settingsColorsSetToDefault: (state, _action: CanvasPayloadAction<unknown>) => {
      state.bgColor = RGBA_BLACK;
      state.fgColor = RGBA_WHITE;
    },
  },
});

export const {
  settingsClipToBboxChanged,
  settingsDynamicGridToggled,
  settingsShowHUDToggled,
  settingsInvertScrollForToolWidthChanged,
  settingsOutputOnlyMaskedRegionsToggled,
  settingsAutoProcessToggled,
  settingsSnapToGridToggled,
  settingsShowProgressOnCanvasToggled,
  settingsBboxOverlayToggled,
  settingsPreserveMaskToggled,
  settingsIsolatedStagingPreviewToggled,
  settingsIsolatedLayerPreviewToggled,
  settingsPressureSensitivityToggled,
  settingsRuleOfThirdsToggled,
  settingsSaveAllImagesToGalleryToggled,
  settingsStagingAreaAutoSwitchChanged,
} = canvasSettingsSlice.actions;

export const {
  settingsBrushWidthChanged,
  settingsEraserWidthChanged,
  settingsActiveColorToggled,
  settingsBgColorChanged,
  settingsFgColorChanged,
  settingsColorsSetToDefault,
} = canvasInstanceSettingsSlice.actions;

const isCanvasInstanceSettingsAction = isAnyOf(...Object.values(canvasInstanceSettingsSlice.actions));

export const canvasSettingsReducer = (state: CanvasSettingsState, action: UnknownAction): CanvasSettingsState => {
  state = canvasSettingsSlice.reducer(state, action);

  if (!isCanvasInstanceSettingsAction(action)) {
    return state;
  }

  const canvasId = action.payload.canvasId;

  return {
    ...state,
    canvases: {
      ...state.canvases,
      [canvasId]: canvasInstanceSettingsSlice.reducer(state.canvases[canvasId], action),
    },
  };
};

export const canvasSettingsSliceConfig: SliceConfig<typeof canvasSettingsSlice> = {
  slice: canvasSettingsSlice,
  schema: zCanvasSettingsState,
  getInitialState: getInitialCanvasSettingsState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      if (!('_version' in state)) {
        // Migrate from v1: slice represented a canvas settings instance -> slice represents multiple canvas settings instances
        const settings = {
          canvasId: MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER,
          ...state,
        } as CanvasInstanceSettings;

        state = {
          _version: 1,
          shared: {
            ...state,
          },
          canvases: { [settings.canvasId]: settings },
        };
      }

      return zCanvasSettingsState.parse(state);
    },
  },
};

export const buildSelectCanvasSettingsByCanvasId = (canvasId: string) =>
  createSelector(
    selectCanvasSharedSettings,
    (state: RootState) => selectCanvasInstanceSettings(state, canvasId),
    (sharedSettings, instanceSettings) => {
      return {
        ...sharedSettings,
        ...instanceSettings,
      };
    }
  );
const selectCanvasSharedSettings = (state: RootState) => state.canvasSettings.shared;
const selectCanvasInstanceSettings = (state: RootState, canvasId: string) => {
  const settings = state.canvasSettings.canvases[canvasId];
  assert(settings, 'Settings must exist for a canvas once the canvas has been created');
  return settings;
};

const buildCanvasSharedSettingsSelector =
  <T>(selector: Selector<CanvasSharedSettings, T>) =>
  (state: RootState) =>
    selector(selectCanvasSharedSettings(state));
const buildCanvasInstanceSettingsSelector =
  <T>(selector: Selector<CanvasInstanceSettings, T>) =>
  (state: RootState, canvasId: string) =>
    selector(selectCanvasInstanceSettings(state, canvasId));

export const selectPreserveMask = buildCanvasSharedSettingsSelector((state) => state.preserveMask);
export const selectOutputOnlyMaskedRegions = buildCanvasSharedSettingsSelector(
  (state) => state.outputOnlyMaskedRegions
);
export const selectDynamicGrid = buildCanvasSharedSettingsSelector((state) => state.dynamicGrid);
export const selectInvertScrollForToolWidth = buildCanvasSharedSettingsSelector(
  (state) => state.invertScrollForToolWidth
);
export const selectBboxOverlay = buildCanvasSharedSettingsSelector((state) => state.bboxOverlay);
export const selectShowHUD = buildCanvasSharedSettingsSelector((state) => state.showHUD);
export const selectClipToBbox = buildCanvasSharedSettingsSelector((state) => state.clipToBbox);
export const selectAutoProcess = buildCanvasSharedSettingsSelector((state) => state.autoProcess);
export const selectSnapToGrid = buildCanvasSharedSettingsSelector((state) => state.snapToGrid);
export const selectShowProgressOnCanvas = buildCanvasSharedSettingsSelector((state) => state.showProgressOnCanvas);
export const selectIsolatedStagingPreview = buildCanvasSharedSettingsSelector((state) => state.isolatedStagingPreview);
export const selectIsolatedLayerPreview = buildCanvasSharedSettingsSelector((state) => state.isolatedLayerPreview);
export const selectPressureSensitivity = buildCanvasSharedSettingsSelector((state) => state.pressureSensitivity);
export const selectRuleOfThirds = buildCanvasSharedSettingsSelector((state) => state.ruleOfThirds);
export const selectSaveAllImagesToGallery = buildCanvasSharedSettingsSelector((state) => state.saveAllImagesToGallery);
export const selectStagingAreaAutoSwitch = buildCanvasSharedSettingsSelector((state) => state.stagingAreaAutoSwitch);
export const selectActiveColor = buildCanvasInstanceSettingsSelector((state) => state.activeColor);
export const selectBgColor = buildCanvasInstanceSettingsSelector((state) => state.bgColor);
export const selectFgColor = buildCanvasInstanceSettingsSelector((state) => state.fgColor);
export const selectBrushWidth = buildCanvasInstanceSettingsSelector((state) => state.brushWidth);
export const selectEraserWidth = buildCanvasInstanceSettingsSelector((state) => state.eraserWidth);
