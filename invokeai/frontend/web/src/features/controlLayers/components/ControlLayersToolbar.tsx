/* eslint-disable i18next/no-literal-string */
import { Button } from '@chakra-ui/react';
import { Flex, Switch } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { BrushWidth } from 'features/controlLayers/components/BrushWidth';
import ControlLayersSettingsPopover from 'features/controlLayers/components/ControlLayersSettingsPopover';
import { EraserWidth } from 'features/controlLayers/components/EraserWidth';
import { FillColorPicker } from 'features/controlLayers/components/FillColorPicker';
import { NewSessionButton } from 'features/controlLayers/components/NewSessionButton';
import { ResetCanvasButton } from 'features/controlLayers/components/ResetCanvasButton';
import { ToolChooser } from 'features/controlLayers/components/ToolChooser';
import { UndoRedoButtonGroup } from 'features/controlLayers/components/UndoRedoButtonGroup';
import { $canvasManager } from 'features/controlLayers/konva/CanvasManager';
import { ToggleProgressButton } from 'features/gallery/components/ImageViewer/ToggleProgressButton';
import { ViewerToggleMenu } from 'features/gallery/components/ImageViewer/ViewerToggleMenu';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';

export const ControlLayersToolbar = memo(() => {
  const tool = useAppSelector((s) => s.canvasV2.tool.selected);
  const canvasManager = useStore($canvasManager);
  const bbox = useCallback(() => {
    if (!canvasManager) {
      return;
    }
    for (const l of canvasManager.layers.values()) {
      l.calculateBbox();
    }
  }, [canvasManager]);
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
        {tool === 'brush' && <BrushWidth />}
        {tool === 'eraser' && <EraserWidth />}
      </Flex>
      <Button onClick={bbox}>bbox</Button>
      <Switch onChange={onChangeDebugging}>debug</Switch>
      <Flex flex={1} justifyContent="center">
        <Flex gap={2} marginInlineStart="auto" alignItems="center">
          <FillColorPicker />
          <UndoRedoButtonGroup />
          <ControlLayersSettingsPopover />
          <ResetCanvasButton />
          <NewSessionButton />
          <ViewerToggleMenu />
        </Flex>
      </Flex>
    </Flex>
  );
});

ControlLayersToolbar.displayName = 'ControlLayersToolbar';
