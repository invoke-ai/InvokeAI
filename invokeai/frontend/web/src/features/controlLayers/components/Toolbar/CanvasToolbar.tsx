import { Divider, Flex } from '@invoke-ai/ui-library';
import { CanvasSettingsPopover } from 'features/controlLayers/components/Settings/CanvasSettingsPopover';
import { ToolColorPicker } from 'features/controlLayers/components/Tool/ToolFillColorPicker';
import { ToolSettings } from 'features/controlLayers/components/Tool/ToolSettings';
import { CanvasToolbarFitBboxToLayersButton } from 'features/controlLayers/components/Toolbar/CanvasToolbarFitBboxToLayersButton';
import { CanvasToolbarNewSessionMenuButton } from 'features/controlLayers/components/Toolbar/CanvasToolbarNewSessionMenuButton';
import { CanvasToolbarRedoButton } from 'features/controlLayers/components/Toolbar/CanvasToolbarRedoButton';
import { CanvasToolbarResetViewButton } from 'features/controlLayers/components/Toolbar/CanvasToolbarResetViewButton';
import { CanvasToolbarSaveToGalleryButton } from 'features/controlLayers/components/Toolbar/CanvasToolbarSaveToGalleryButton';
import { CanvasToolbarScale } from 'features/controlLayers/components/Toolbar/CanvasToolbarScale';
import { CanvasToolbarUndoButton } from 'features/controlLayers/components/Toolbar/CanvasToolbarUndoButton';
import { useCanvasDeleteLayerHotkey } from 'features/controlLayers/hooks/useCanvasDeleteLayerHotkey';
import { useCanvasEntityQuickSwitchHotkey } from 'features/controlLayers/hooks/useCanvasEntityQuickSwitchHotkey';
import { useCanvasFilterHotkey } from 'features/controlLayers/hooks/useCanvasFilterHotkey';
import { useCanvasResetLayerHotkey } from 'features/controlLayers/hooks/useCanvasResetLayerHotkey';
import { useCanvasSnapToGridHotkey } from 'features/controlLayers/hooks/useCanvasSnapToGridHotkey';
import { useCanvasToggleNonRasterLayersHotkey } from 'features/controlLayers/hooks/useCanvasToggleNonRasterLayersHotkey';
import { useCanvasTransformHotkey } from 'features/controlLayers/hooks/useCanvasTransformHotkey';
import { useCanvasUndoRedoHotkeys } from 'features/controlLayers/hooks/useCanvasUndoRedoHotkeys';
import { useNextPrevEntityHotkeys } from 'features/controlLayers/hooks/useNextPrevEntity';
import { memo } from 'react';

import { CanvasToolbarSnappingToolButton } from './CanvasToolbarSnappingToolButton';

export const CanvasToolbar = memo(() => {
  useCanvasResetLayerHotkey();
  useCanvasDeleteLayerHotkey();
  useCanvasUndoRedoHotkeys();
  useCanvasEntityQuickSwitchHotkey();
  useNextPrevEntityHotkeys();
  useCanvasTransformHotkey();
  useCanvasFilterHotkey();
  useCanvasToggleNonRasterLayersHotkey();
  useCanvasSnapToGridHotkey();

  return (
    <Flex w="full" gap={2} alignItems="center" px={2}>
      <ToolColorPicker />
      <ToolSettings />
      <Flex alignItems="center" h="full" flexGrow={1} justifyContent="flex-end">
        <CanvasToolbarScale />
        <CanvasToolbarResetViewButton />
        <CanvasToolbarFitBboxToLayersButton />
        <CanvasToolbarSnappingToolButton />
      </Flex>
      <Divider orientation="vertical" />
      <Flex alignItems="center" h="full">
        <CanvasToolbarSaveToGalleryButton />
        <CanvasToolbarUndoButton />
        <CanvasToolbarRedoButton />
        <CanvasToolbarNewSessionMenuButton />
        <CanvasSettingsPopover />
      </Flex>
    </Flex>
  );
});

CanvasToolbar.displayName = 'CanvasToolbar';
