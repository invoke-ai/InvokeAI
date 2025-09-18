import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
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

const zCanvasSharedSettingsState = z.object({
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
type CanvasSharedSettingsState = z.infer<typeof zCanvasSharedSettingsState>;

const zCanvasInstanceSettingsState = z.object({
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
type CanvasInstanceSettingsState = z.infer<typeof zCanvasInstanceSettingsState>;

const zCanvasSettingsState = z.object({
  _version: z.literal(1),
  shared: zCanvasSharedSettingsState,
  canvases: z.array(zCanvasInstanceSettingsState),
});
type CanvasSettingsState = z.infer<typeof zCanvasSettingsState>;

const getInitialCanvasSharedSettingsState = (): CanvasSharedSettingsState => ({
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
const getInitialCanvasInstanceSettingsState = (canvasId: string): CanvasInstanceSettingsState => ({
  canvasId,
  brushWidth: 50,
  eraserWidth: 50,
  activeColor: 'fgColor',
  bgColor: RGBA_BLACK,
  fgColor: RGBA_WHITE,
});
const getInitialState = (): CanvasSettingsState => ({
  _version: 1,
  shared: getInitialCanvasSharedSettingsState(),
  canvases: [],
});

type CanvasPayload<T> = { canvasId: string } & T;
type CanvasPayloadAction<T> = PayloadAction<CanvasPayload<T>>;

const slice = createSlice({
  name: 'canvasSettings',
  initialState: getInitialState(),
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
    settingsBrushWidthChanged: (state, action: CanvasPayloadAction<{ brushWidth: number }>) => {
      const { canvasId, brushWidth } = action.payload;

      const settings = state.canvases.find((settings) => settings.canvasId === canvasId);
      if (!settings) {
        return;
      }

      settings.brushWidth = Math.round(brushWidth);
    },
    settingsEraserWidthChanged: (state, action: CanvasPayloadAction<{ eraserWidth: number }>) => {
      const { canvasId, eraserWidth } = action.payload;

      const settings = state.canvases.find((settings) => settings.canvasId === canvasId);
      if (!settings) {
        return;
      }

      settings.eraserWidth = Math.round(eraserWidth);
    },
    settingsActiveColorToggled: (state, action: CanvasPayloadAction<unknown>) => {
      const { canvasId } = action.payload;

      const settings = state.canvases.find((settings) => settings.canvasId === canvasId);
      if (!settings) {
        return;
      }

      settings.activeColor = settings.activeColor === 'bgColor' ? 'fgColor' : 'bgColor';
    },
    settingsBgColorChanged: (state, action: CanvasPayloadAction<{ bgColor: Partial<RgbaColor> }>) => {
      const { canvasId, bgColor } = action.payload;

      const settings = state.canvases.find((settings) => settings.canvasId === canvasId);
      if (!settings) {
        return;
      }

      settings.bgColor = { ...settings.bgColor, ...bgColor };
    },
    settingsFgColorChanged: (state, action: CanvasPayloadAction<{ fgColor: Partial<RgbaColor> }>) => {
      const { canvasId, fgColor } = action.payload;

      const settings = state.canvases.find((settings) => settings.canvasId === canvasId);
      if (!settings) {
        return;
      }

      settings.fgColor = { ...settings.fgColor, ...fgColor };
    },
    settingsColorsSetToDefault: (state, action: CanvasPayloadAction<unknown>) => {
      const { canvasId } = action.payload;

      const settings = state.canvases.find((settings) => settings.canvasId === canvasId);
      if (!settings) {
        return;
      }

      settings.bgColor = RGBA_BLACK;
      settings.fgColor = RGBA_WHITE;
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
      action: PayloadAction<{ stagingAreaAutoSwitch: CanvasSharedSettingsState['stagingAreaAutoSwitch'] }>
    ) => {
      const { stagingAreaAutoSwitch } = action.payload;

      state.shared.stagingAreaAutoSwitch = stagingAreaAutoSwitch;
    },
  },
  extraReducers(builder) {
    builder.addCase(canvasCreated, (state, action) => {
      const canvasSettings = getInitialCanvasInstanceSettingsState(action.payload.id);
      state.canvases.push(canvasSettings);
    });
    builder.addCase(canvasRemoved, (state, action) => {
      state.canvases = state.canvases.filter((settings) => settings.canvasId !== action.payload.id);
    });
    builder.addCase(canvasMultiCanvasMigrated, (state, action) => {
      const settings = state.canvases.find((settings) => settings.canvasId === MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER);
      if (!settings) {
        return;
      }
      settings.canvasId = action.payload.id;
    });
  },
});

export const {
  settingsClipToBboxChanged,
  settingsDynamicGridToggled,
  settingsShowHUDToggled,
  settingsBrushWidthChanged,
  settingsEraserWidthChanged,
  settingsActiveColorToggled,
  settingsBgColorChanged,
  settingsFgColorChanged,
  settingsColorsSetToDefault,
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
} = slice.actions;

export const canvasSettingsSliceConfig: SliceConfig<typeof slice> = {
  slice,
  schema: zCanvasSettingsState,
  getInitialState,
  persistConfig: {
    migrate: (state) => {
      assert(isPlainObject(state));
      if (!('_version' in state)) {
        // Migrate from v1: slice represented a canvas settings instance -> slice represents multiple canvas settings instances
        const canvas = {
          canvasId: MIGRATION_MULTI_CANVAS_ID_PLACEHOLDER,
          ...state,
        } as CanvasInstanceSettingsState;

        state = {
          _version: 1,
          shared: {
            ...state,
          },
          canvases: [canvas],
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
  const settings = state.canvasSettings.canvases.find((settings) => settings.canvasId === canvasId);
  assert(settings, 'Settings must exist for a canvas once the canvas has been created');
  return settings;
};

const buildCanvasSharedSettingsSelector =
  <T>(selector: Selector<CanvasSharedSettingsState, T>) =>
  (state: RootState) =>
    selector(selectCanvasSharedSettings(state));
const buildCanvasInstanceSettingsSelector =
  <T>(selector: Selector<CanvasInstanceSettingsState, T>) =>
  (state: RootState, canvasId: string) =>
    selector(selectCanvasInstanceSettings(state, canvasId));

export const selectPreserveMask = buildCanvasSharedSettingsSelector((settings) => settings.preserveMask);
export const selectOutputOnlyMaskedRegions = buildCanvasSharedSettingsSelector(
  (settings) => settings.outputOnlyMaskedRegions
);
export const selectDynamicGrid = buildCanvasSharedSettingsSelector((settings) => settings.dynamicGrid);
export const selectInvertScrollForToolWidth = buildCanvasSharedSettingsSelector(
  (settings) => settings.invertScrollForToolWidth
);
export const selectBboxOverlay = buildCanvasSharedSettingsSelector((settings) => settings.bboxOverlay);
export const selectShowHUD = buildCanvasSharedSettingsSelector((settings) => settings.showHUD);
export const selectClipToBbox = buildCanvasSharedSettingsSelector((settings) => settings.clipToBbox);
export const selectAutoProcess = buildCanvasSharedSettingsSelector((settings) => settings.autoProcess);
export const selectSnapToGrid = buildCanvasSharedSettingsSelector((settings) => settings.snapToGrid);
export const selectShowProgressOnCanvas = buildCanvasSharedSettingsSelector(
  (settings) => settings.showProgressOnCanvas
);
export const selectIsolatedStagingPreview = buildCanvasSharedSettingsSelector(
  (settings) => settings.isolatedStagingPreview
);
export const selectIsolatedLayerPreview = buildCanvasSharedSettingsSelector(
  (settings) => settings.isolatedLayerPreview
);
export const selectPressureSensitivity = buildCanvasSharedSettingsSelector((settings) => settings.pressureSensitivity);
export const selectRuleOfThirds = buildCanvasSharedSettingsSelector((settings) => settings.ruleOfThirds);
export const selectSaveAllImagesToGallery = buildCanvasSharedSettingsSelector(
  (settings) => settings.saveAllImagesToGallery
);
export const selectStagingAreaAutoSwitch = buildCanvasSharedSettingsSelector(
  (settings) => settings.stagingAreaAutoSwitch
);
export const selectActiveColor = buildCanvasInstanceSettingsSelector((settings) => settings.activeColor);
export const selectBgColor = buildCanvasInstanceSettingsSelector((settings) => settings.bgColor);
export const selectFgColor = buildCanvasInstanceSettingsSelector((settings) => settings.fgColor);
export const selectBrushWidth = buildCanvasInstanceSettingsSelector((settings) => settings.brushWidth);
export const selectEraserWidth = buildCanvasInstanceSettingsSelector((settings) => settings.eraserWidth);
