import type { PayloadAction, Selector } from '@reduxjs/toolkit';
import { createSlice, isAnyOf } from '@reduxjs/toolkit';
import type { RootState } from 'app/store/store';
import type { CanvasSettingsState, RgbaColor } from 'features/controlLayers/store/types';
import { RGBA_BLACK, RGBA_WHITE } from 'features/controlLayers/store/types';
import { assert } from 'tsafe';

export const getInitialCanvasSettings = (): CanvasSettingsState => ({
  showHUD: true,
  clipToBbox: false,
  dynamicGrid: false,
  invertScrollForToolWidth: false,
  brushWidth: 50,
  eraserWidth: 50,
  activeColor: 'fgColor',
  bgColor: RGBA_BLACK,
  fgColor: RGBA_WHITE,
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

export const canvasSettingsState = createSlice({
  name: 'canvasSettings',
  initialState: {} as CanvasSettingsState,
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
    settingsActiveColorToggled: (state) => {
      state.activeColor = state.activeColor === 'bgColor' ? 'fgColor' : 'bgColor';
    },
    settingsBgColorChanged: (state, action: PayloadAction<Partial<RgbaColor>>) => {
      state.bgColor = { ...state.bgColor, ...action.payload };
    },
    settingsFgColorChanged: (state, action: PayloadAction<Partial<RgbaColor>>) => {
      state.fgColor = { ...state.fgColor, ...action.payload };
    },
    settingsColorsSetToDefault: (state) => {
      state.bgColor = RGBA_BLACK;
      state.fgColor = RGBA_WHITE;
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

export const isCanvasSettingsStateAction = isAnyOf(...Object.values(canvasSettingsState.actions));

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
} = canvasSettingsState.actions;

export const selectCanvasSettingsByCanvasId = (state: RootState, canvasId: string) => {
  const instance = state.canvas.canvases[canvasId];
  assert(instance, 'Canvas does not exist');
  return instance.settings;
};
const selectActiveCanvasSettings = (state: RootState) => {
  return state.canvas.canvases[state.canvas.activeCanvasId]!.settings;
};

const buildActiveCanvasSettingsSelector =
  <T>(selector: Selector<CanvasSettingsState, T>) =>
  (state: RootState) =>
    selector(selectActiveCanvasSettings(state));

export const selectPreserveMask = buildActiveCanvasSettingsSelector((state) => state.preserveMask);
export const selectOutputOnlyMaskedRegions = buildActiveCanvasSettingsSelector(
  (state) => state.outputOnlyMaskedRegions
);
export const selectDynamicGrid = buildActiveCanvasSettingsSelector((state) => state.dynamicGrid);
export const selectInvertScrollForToolWidth = buildActiveCanvasSettingsSelector(
  (state) => state.invertScrollForToolWidth
);
export const selectBboxOverlay = buildActiveCanvasSettingsSelector((state) => state.bboxOverlay);
export const selectShowHUD = buildActiveCanvasSettingsSelector((state) => state.showHUD);
export const selectClipToBbox = buildActiveCanvasSettingsSelector((state) => state.clipToBbox);
export const selectAutoProcess = buildActiveCanvasSettingsSelector((state) => state.autoProcess);
export const selectSnapToGrid = buildActiveCanvasSettingsSelector((state) => state.snapToGrid);
export const selectShowProgressOnCanvas = buildActiveCanvasSettingsSelector((state) => state.showProgressOnCanvas);
export const selectIsolatedStagingPreview = buildActiveCanvasSettingsSelector((state) => state.isolatedStagingPreview);
export const selectIsolatedLayerPreview = buildActiveCanvasSettingsSelector((state) => state.isolatedLayerPreview);
export const selectPressureSensitivity = buildActiveCanvasSettingsSelector((state) => state.pressureSensitivity);
export const selectRuleOfThirds = buildActiveCanvasSettingsSelector((state) => state.ruleOfThirds);
export const selectSaveAllImagesToGallery = buildActiveCanvasSettingsSelector((state) => state.saveAllImagesToGallery);
export const selectStagingAreaAutoSwitch = buildActiveCanvasSettingsSelector((state) => state.stagingAreaAutoSwitch);
export const selectActiveColor = buildActiveCanvasSettingsSelector((state) => state.activeColor);
export const selectBgColor = buildActiveCanvasSettingsSelector((state) => state.bgColor);
export const selectFgColor = buildActiveCanvasSettingsSelector((state) => state.fgColor);
export const selectBrushWidth = buildActiveCanvasSettingsSelector((state) => state.brushWidth);
export const selectEraserWidth = buildActiveCanvasSettingsSelector((state) => state.eraserWidth);
