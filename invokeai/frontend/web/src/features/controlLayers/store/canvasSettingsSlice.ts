import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { SliceConfig } from 'app/store/types';
import { zRgbaColor } from 'features/controlLayers/store/types';
import { z } from 'zod';

const zAutoSwitchMode = z.enum(['off', 'switch_on_start', 'switch_on_finish']);
export type AutoSwitchMode = z.infer<typeof zAutoSwitchMode>;

const zCanvasSettingsState = z.object({
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
   * The width of the brush tool.
   */
  brushWidth: z.int().gt(0),
  /**
   * The width of the eraser tool.
   */
  eraserWidth: z.int().gt(0),
  /**
   * The color to use when drawing lines or filling shapes.
   */
  color: zRgbaColor,
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

type CanvasSettingsState = z.infer<typeof zCanvasSettingsState>;
const getInitialState = (): CanvasSettingsState => ({
  showHUD: true,
  clipToBbox: false,
  dynamicGrid: false,
  invertScrollForToolWidth: false,
  brushWidth: 50,
  eraserWidth: 50,
  color: { r: 31, g: 160, b: 224, a: 1 }, // invokeBlue.500
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

const slice = createSlice({
  name: 'canvasSettings',
  initialState: getInitialState(),
  reducers: {
    settingsClipToBboxChanged: (state, action: PayloadAction<CanvasSettingsState['clipToBbox']>) => {
      state.clipToBbox = action.payload;
    },
    settingsDynamicGridToggled: (state) => {
      state.dynamicGrid = !state.dynamicGrid;
    },
    settingsShowHUDToggled: (state) => {
      state.showHUD = !state.showHUD;
    },
    settingsBrushWidthChanged: (state, action: PayloadAction<CanvasSettingsState['brushWidth']>) => {
      state.brushWidth = Math.round(action.payload);
    },
    settingsEraserWidthChanged: (state, action: PayloadAction<CanvasSettingsState['eraserWidth']>) => {
      state.eraserWidth = Math.round(action.payload);
    },
    settingsColorChanged: (state, action: PayloadAction<Partial<CanvasSettingsState['color']>>) => {
      state.color = { ...state.color, ...action.payload };
    },
    settingsInvertScrollForToolWidthChanged: (
      state,
      action: PayloadAction<CanvasSettingsState['invertScrollForToolWidth']>
    ) => {
      state.invertScrollForToolWidth = action.payload;
    },
    settingsOutputOnlyMaskedRegionsToggled: (state) => {
      state.outputOnlyMaskedRegions = !state.outputOnlyMaskedRegions;
    },
    settingsAutoProcessToggled: (state) => {
      state.autoProcess = !state.autoProcess;
    },
    settingsSnapToGridToggled: (state) => {
      state.snapToGrid = !state.snapToGrid;
    },
    settingsShowProgressOnCanvasToggled: (state) => {
      state.showProgressOnCanvas = !state.showProgressOnCanvas;
    },
    settingsBboxOverlayToggled: (state) => {
      state.bboxOverlay = !state.bboxOverlay;
    },
    settingsPreserveMaskToggled: (state) => {
      state.preserveMask = !state.preserveMask;
    },
    settingsIsolatedStagingPreviewToggled: (state) => {
      state.isolatedStagingPreview = !state.isolatedStagingPreview;
    },
    settingsIsolatedLayerPreviewToggled: (state) => {
      state.isolatedLayerPreview = !state.isolatedLayerPreview;
    },
    settingsPressureSensitivityToggled: (state) => {
      state.pressureSensitivity = !state.pressureSensitivity;
    },
    settingsRuleOfThirdsToggled: (state) => {
      state.ruleOfThirds = !state.ruleOfThirds;
    },
    settingsSaveAllImagesToGalleryToggled: (state) => {
      state.saveAllImagesToGallery = !state.saveAllImagesToGallery;
    },
    settingsStagingAreaAutoSwitchChanged: (
      state,
      action: PayloadAction<CanvasSettingsState['stagingAreaAutoSwitch']>
    ) => {
      state.stagingAreaAutoSwitch = action.payload;
    },
  },
});

export const {
  settingsClipToBboxChanged,
  settingsDynamicGridToggled,
  settingsShowHUDToggled,
  settingsBrushWidthChanged,
  settingsEraserWidthChanged,
  settingsColorChanged,
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
    migrate: (state) => zCanvasSettingsState.parse(state),
  },
};

export const selectCanvasSettingsSlice = (s: RootState) => s.canvasSettings;
const createCanvasSettingsSelector = <T>(selector: Selector<CanvasSettingsState, T>) =>
  createSelector(selectCanvasSettingsSlice, selector);

export const selectPreserveMask = createCanvasSettingsSelector((settings) => settings.preserveMask);
export const selectOutputOnlyMaskedRegions = createCanvasSettingsSelector(
  (settings) => settings.outputOnlyMaskedRegions
);
export const selectDynamicGrid = createCanvasSettingsSelector((settings) => settings.dynamicGrid);
export const selectBboxOverlay = createCanvasSettingsSelector((settings) => settings.bboxOverlay);
export const selectShowHUD = createCanvasSettingsSelector((settings) => settings.showHUD);
export const selectAutoProcess = createCanvasSettingsSelector((settings) => settings.autoProcess);
export const selectSnapToGrid = createCanvasSettingsSelector((settings) => settings.snapToGrid);
export const selectShowProgressOnCanvas = createCanvasSettingsSelector(
  (canvasSettings) => canvasSettings.showProgressOnCanvas
);
export const selectIsolatedStagingPreview = createCanvasSettingsSelector((settings) => settings.isolatedStagingPreview);
export const selectIsolatedLayerPreview = createCanvasSettingsSelector((settings) => settings.isolatedLayerPreview);
export const selectPressureSensitivity = createCanvasSettingsSelector((settings) => settings.pressureSensitivity);
export const selectRuleOfThirds = createCanvasSettingsSelector((settings) => settings.ruleOfThirds);
export const selectSaveAllImagesToGallery = createCanvasSettingsSelector((settings) => settings.saveAllImagesToGallery);
export const selectStagingAreaAutoSwitch = createCanvasSettingsSelector((settings) => settings.stagingAreaAutoSwitch);
