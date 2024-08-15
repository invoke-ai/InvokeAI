/* eslint-disable i18next/no-literal-string */
import { Flex, Switch } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasResetViewButton } from 'features/controlLayers/components/CanvasResetViewButton';
import { CanvasScale } from 'features/controlLayers/components/CanvasScale';
import ControlLayersSettingsPopover from 'features/controlLayers/components/ControlLayersSettingsPopover';
import { ResetCanvasButton } from 'features/controlLayers/components/ResetCanvasButton';
import { ToolBrushWidth } from 'features/controlLayers/components/Tool/ToolBrushWidth';
import { ToolChooser } from 'features/controlLayers/components/Tool/ToolChooser';
import { ToolEraserWidth } from 'features/controlLayers/components/Tool/ToolEraserWidth';
import { ToolFillColorPicker } from 'features/controlLayers/components/Tool/ToolFillColorPicker';
import { UndoRedoButtonGroup } from 'features/controlLayers/components/UndoRedoButtonGroup';
import { $canvasManager } from 'features/controlLayers/konva/CanvasManager';
import { ToggleProgressButton } from 'features/gallery/components/ImageViewer/ToggleProgressButton';
import { ViewerToggleMenu } from 'features/gallery/components/ImageViewer/ViewerToggleMenu';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';

export const ControlLayersToolbar = memo(() => {
  const tool = useAppSelector((s) => s.canvasV2.tool.selected);
  const canvasManager = useStore($canvasManager);
  const onChangeDebugging = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      if (!canvasManager) {
        return;
      }
      if (e.target.checked) {
        canvasManager.enableDebugging();
      } else {
        canvasManager.disableDebugging();
      }
    },
    [canvasManager]
  );
  return (
    <Flex w="full" gap={2}>
      <Flex flex={1} justifyContent="center">
        <Flex gap={2} marginInlineEnd="auto" alignItems="center">
          <ToggleProgressButton />
          <ToolChooser />
        </Flex>
      </Flex>
      <Flex flex={1} gap={2} justifyContent="center" alignItems="center">
        {tool === 'brush' && <ToolBrushWidth />}
        {tool === 'eraser' && <ToolEraserWidth />}
      </Flex>
      <CanvasScale />
      <CanvasResetViewButton />
      <Switch onChange={onChangeDebugging}>debug</Switch>
      <Flex flex={1} justifyContent="center">
        <Flex gap={2} marginInlineStart="auto" alignItems="center">
          <ToolFillColorPicker />
          <UndoRedoButtonGroup />
          <ControlLayersSettingsPopover />
          <ResetCanvasButton />
          <ViewerToggleMenu />
        </Flex>
      </Flex>
    </Flex>
  );
});

ControlLayersToolbar.displayName = 'ControlLayersToolbar';
