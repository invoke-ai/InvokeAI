/* eslint-disable i18next/no-literal-string */
import { Divider, Flex, Heading } from '@invoke-ai/ui-library';
import { CanvasSettingsPopover } from 'features/controlLayers/components/Settings/CanvasSettingsPopover';
import { StartOverButton } from 'features/controlLayers/components/StartOverButton';
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
import { useCanvasTransformHotkey } from 'features/controlLayers/hooks/useCanvasTransformHotkey';
import { useCanvasUndoRedoHotkeys } from 'features/controlLayers/hooks/useCanvasUndoRedoHotkeys';
import { useNextPrevEntityHotkeys } from 'features/controlLayers/hooks/useNextPrevEntity';
import { memo } from 'react';

export const CanvasToolbar = memo(() => {
  useCanvasResetLayerHotkey();
  useCanvasDeleteLayerHotkey();
  useCanvasUndoRedoHotkeys();
  useCanvasEntityQuickSwitchHotkey();
  useNextPrevEntityHotkeys();
  useCanvasTransformHotkey();
  useCanvasFilterHotkey();

  return (
    <Flex w="full" gap={2} alignItems="center" px={2}>
      <Heading size="sm" me={2}>
        Canvas
      </Heading>
      <Divider orientation="vertical" />
      <ToolColorPicker />
      <ToolSettings />
      <Flex alignItems="center" h="full" flexGrow={1} justifyContent="flex-end">
        <CanvasToolbarScale />
        <CanvasToolbarResetViewButton />
        <CanvasToolbarFitBboxToLayersButton />
      </Flex>
      <Divider orientation="vertical" />
      <Flex alignItems="center" h="full">
        <CanvasToolbarSaveToGalleryButton />
        <CanvasToolbarUndoButton />
        <CanvasToolbarRedoButton />
        <CanvasToolbarNewSessionMenuButton />
        <CanvasSettingsPopover />
      </Flex>
      <Divider orientation="vertical" />
      <StartOverButton />
    </Flex>
  );
});

CanvasToolbar.displayName = 'CanvasToolbar';
