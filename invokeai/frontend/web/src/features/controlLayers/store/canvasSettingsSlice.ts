import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSelector, createSlice } from '@reduxjs/toolkit';
import type { PersistConfig, RootState } from 'app/store/store';
import { newCanvasSessionRequested, newGallerySessionRequested } from 'features/controlLayers/store/actions';
import type { RgbaColor } from 'features/controlLayers/store/types';

type CanvasSettingsState = {
  /**
   * Whether to show HUD (Heads-Up Display) on the canvas.
   */
  showHUD: boolean;
  /**
   * Whether to clip lines and shapes to the generation bounding box. If disabled, lines and shapes will be clipped to
   * the canvas bounds.
   */
  clipToBbox: boolean;
  /**
   * Whether to show a dynamic grid on the canvas. If disabled, a checkerboard pattern will be shown instead.
   */
  dynamicGrid: boolean;
  /**
   * Whether to invert the scroll direction when adjusting the brush or eraser width with the scroll wheel.
   */
  invertScrollForToolWidth: boolean;
  /**
   * The width of the brush tool.
   */
  brushWidth: number;
  /**
   * The width of the eraser tool.
   */
  eraserWidth: number;
  /**
   * The color to use when drawing lines or filling shapes.
   */
  color: RgbaColor;
  /**
   * Whether to send generated images to canvas staging area. When disabled, generated images will be sent directly to
   * the gallery.
   */
  sendToCanvas: boolean;
  /**
   * Whether to composite inpainted/outpainted regions back onto the source image when saving canvas generations.
   *
   * If disabled, inpainted/outpainted regions will be saved with a transparent background.
   *
   * When `sendToCanvas` is disabled, this setting is ignored, masked regions will always be composited.
   */
  outputOnlyMaskedRegions: boolean;
  /**
   * Whether to automatically process the filter when the filter configuration changes.
   */
  autoProcessFilter: boolean;
  /**
   * The snap-to-grid setting for the canvas.
   */
  snapToGrid: boolean;
  /**
   * Whether to show progress on the canvas when generating images.
   */
  showProgressOnCanvas: boolean;
  /**
   * Whether to show the bounding box overlay on the canvas.
   */
  bboxOverlay: boolean;
  /**
   * Whether to preserve the masked region instead of inpainting it.
   */
  preserveMask: boolean;
  /**
   * Whether to show only raster layers while staging.
   */
  isolatedStagingPreview: boolean;
  /**
   * Whether to show only the selected layer while filtering.
   */
  isolatedFilteringPreview: boolean;
  /**
   * Whether to show only the selected layer while transforming.
   */
  isolatedTransformingPreview: boolean;
  /**
   * Whether to use pressure sensitivity for the brush and eraser tool when a pen device is used.
   */
  pressureSensitivity: boolean;
};

const initialState: CanvasSettingsState = {
  showHUD: true,
  clipToBbox: false,
  dynamicGrid: false,
  brushWidth: 50,
  eraserWidth: 50,
  invertScrollForToolWidth: false,
  color: { r: 31, g: 160, b: 224, a: 1 }, // invokeBlue.500
  sendToCanvas: false,
  outputOnlyMaskedRegions: false,
  autoProcessFilter: true,
  snapToGrid: true,
  showProgressOnCanvas: true,
  bboxOverlay: false,
  preserveMask: false,
  isolatedStagingPreview: true,
  isolatedFilteringPreview: true,
  isolatedTransformingPreview: true,
  pressureSensitivity: true,
};

export const canvasSettingsSlice = createSlice({
  name: 'canvasSettings',
  initialState,
  reducers: {
    settingsClipToBboxChanged: (state, action: PayloadAction<boolean>) => {
      state.clipToBbox = action.payload;
    },
    settingsDynamicGridToggled: (state) => {
      state.dynamicGrid = !state.dynamicGrid;
    },
    settingsShowHUDToggled: (state) => {
      state.showHUD = !state.showHUD;
    },
    settingsBrushWidthChanged: (state, action: PayloadAction<number>) => {
      state.brushWidth = Math.round(action.payload);
    },
    settingsEraserWidthChanged: (state, action: PayloadAction<number>) => {
      state.eraserWidth = Math.round(action.payload);
    },
    settingsColorChanged: (state, action: PayloadAction<RgbaColor>) => {
      state.color = action.payload;
    },
    settingsInvertScrollForToolWidthChanged: (state, action: PayloadAction<boolean>) => {
      state.invertScrollForToolWidth = action.payload;
    },
    settingsSendToCanvasChanged: (state, action: PayloadAction<boolean>) => {
      state.sendToCanvas = action.payload;
    },
    settingsOutputOnlyMaskedRegionsToggled: (state) => {
      state.outputOnlyMaskedRegions = !state.outputOnlyMaskedRegions;
    },
    settingsAutoProcessFilterToggled: (state) => {
      state.autoProcessFilter = !state.autoProcessFilter;
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
    settingsIsolatedFilteringPreviewToggled: (state) => {
      state.isolatedFilteringPreview = !state.isolatedFilteringPreview;
    },
    settingsIsolatedTransformingPreviewToggled: (state) => {
      state.isolatedTransformingPreview = !state.isolatedTransformingPreview;
    },
    settingsPressureSensitivityToggled: (state) => {
      state.pressureSensitivity = !state.pressureSensitivity;
    },
  },
  extraReducers(builder) {
    builder.addCase(newGallerySessionRequested, (state) => {
      state.sendToCanvas = false;
    });
    builder.addCase(newCanvasSessionRequested, (state) => {
      state.sendToCanvas = true;
    });
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
  settingsSendToCanvasChanged,
  settingsOutputOnlyMaskedRegionsToggled,
  settingsAutoProcessFilterToggled,
  settingsSnapToGridToggled,
  settingsShowProgressOnCanvasToggled,
  settingsBboxOverlayToggled,
  settingsPreserveMaskToggled,
  settingsIsolatedStagingPreviewToggled,
  settingsIsolatedFilteringPreviewToggled,
  settingsIsolatedTransformingPreviewToggled,
  settingsPressureSensitivityToggled,
} = canvasSettingsSlice.actions;

/* eslint-disable-next-line @typescript-eslint/no-explicit-any */
const migrate = (state: any): any => {
  return state;
};

export const canvasSettingsPersistConfig: PersistConfig<CanvasSettingsState> = {
  name: canvasSettingsSlice.name,
  initialState,
  migrate,
  persistDenylist: [],
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
export const selectAutoProcessFilter = createCanvasSettingsSelector((settings) => settings.autoProcessFilter);
export const selectSnapToGrid = createCanvasSettingsSelector((settings) => settings.snapToGrid);
export const selectSendToCanvas = createCanvasSettingsSelector((canvasSettings) => canvasSettings.sendToCanvas);
export const selectShowProgressOnCanvas = createCanvasSettingsSelector(
  (canvasSettings) => canvasSettings.showProgressOnCanvas
);
export const selectIsolatedStagingPreview = createCanvasSettingsSelector((settings) => settings.isolatedStagingPreview);
export const selectIsolatedFilteringPreview = createCanvasSettingsSelector(
  (settings) => settings.isolatedFilteringPreview
);
export const selectIsolatedTransformingPreview = createCanvasSettingsSelector(
  (settings) => settings.isolatedTransformingPreview
);
export const selectPressureSensitivity = createCanvasSettingsSelector((settings) => settings.pressureSensitivity);
