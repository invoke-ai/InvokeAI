/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { BrushColorPicker } from 'features/controlLayers/components/BrushColorPicker';
import { BrushWidth } from 'features/controlLayers/components/BrushSize';
import ControlLayersSettingsPopover from 'features/controlLayers/components/ControlLayersSettingsPopover';
import { ToolChooser } from 'features/controlLayers/components/ToolChooser';
import { UndoRedoButtonGroup } from 'features/controlLayers/components/UndoRedoButtonGroup';
import { $tool } from 'features/controlLayers/store/controlLayersSlice';
import { ToggleProgressButton } from 'features/gallery/components/ImageViewer/ToggleProgressButton';
import { ViewerToggleMenu } from 'features/gallery/components/ImageViewer/ViewerToggleMenu';
import { memo, useMemo } from 'react';

export const ControlLayersToolbar = memo(() => {
  const tool = useStore($tool);
  const withBrushSize = useMemo(() => {
    return tool === 'brush' || tool === 'eraser';
  }, [tool]);
  const withBrushColor = useMemo(() => {
    return tool === 'brush' || tool === 'rect';
  }, [tool]);
  return (
    <Flex w="full" gap={2}>
      <Flex flex={1} justifyContent="center">
        <Flex gap={2} marginInlineEnd="auto">
          <ToggleProgressButton />
          <ToolChooser />
        </Flex>
      </Flex>
      <Flex flex={1} gap={2} justifyContent="center" alignItems="center">
        {withBrushSize && <BrushWidth />}
        {withBrushColor && <BrushColorPicker />}
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
