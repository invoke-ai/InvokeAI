/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { BrushSize } from 'features/controlLayers/components/BrushSize';
import ControlLayersSettingsPopover from 'features/controlLayers/components/ControlLayersSettingsPopover';
import { ToolChooser } from 'features/controlLayers/components/ToolChooser';
import { UndoRedoButtonGroup } from 'features/controlLayers/components/UndoRedoButtonGroup';
import { ViewerToggleMenu } from 'features/gallery/components/ImageViewer/ViewerToggleMenu';
import { memo } from 'react';

export const ControlLayersToolbar = memo(() => {
  return (
    <Flex w="full" gap={2}>
      <Flex flex={1} justifyContent="center">
        <Flex gap={2} marginInlineEnd="auto" />
      </Flex>
      <Flex flex={1} gap={2} justifyContent="center">
        <BrushSize />
        <ToolChooser />
        <UndoRedoButtonGroup />
        <ControlLayersSettingsPopover />
      </Flex>
      <Flex flex={1} justifyContent="center">
        <Flex gap={2} marginInlineStart="auto">
          <ViewerToggleMenu />
        </Flex>
      </Flex>
    </Flex>
  );
});

ControlLayersToolbar.displayName = 'ControlLayersToolbar';
