/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { BrushWidth } from 'features/controlLayers/components/BrushWidth';
import ControlLayersSettingsPopover from 'features/controlLayers/components/ControlLayersSettingsPopover';
import { EraserWidth } from 'features/controlLayers/components/EraserWidth';
import { FillColorPicker } from 'features/controlLayers/components/FillColorPicker';
import { ToolChooser } from 'features/controlLayers/components/ToolChooser';
import { UndoRedoButtonGroup } from 'features/controlLayers/components/UndoRedoButtonGroup';
import { ToggleProgressButton } from 'features/gallery/components/ImageViewer/ToggleProgressButton';
import { ViewerToggleMenu } from 'features/gallery/components/ImageViewer/ViewerToggleMenu';
import { memo } from 'react';

export const ControlLayersToolbar = memo(() => {
  const tool = useAppSelector((s) => s.canvasV2.tool.selected);
  return (
    <Flex w="full" gap={2}>
      <Flex flex={1} justifyContent="center">
        <Flex gap={2} marginInlineEnd="auto">
          <ToggleProgressButton />
          <ToolChooser />
        </Flex>
      </Flex>
      <Flex flex={1} gap={2} justifyContent="center" alignItems="center">
        {tool === 'brush' && <BrushWidth />}
        {tool === 'eraser' && <EraserWidth />}
        <FillColorPicker />
      </Flex>
      <Flex flex={1} justifyContent="center">
        <Flex gap={2} marginInlineStart="auto">
          <UndoRedoButtonGroup />
          <ControlLayersSettingsPopover />
          <ViewerToggleMenu />
        </Flex>
      </Flex>
    </Flex>
  );
});

ControlLayersToolbar.displayName = 'ControlLayersToolbar';
