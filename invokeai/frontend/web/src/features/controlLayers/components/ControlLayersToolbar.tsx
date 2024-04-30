/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { BrushSize } from 'features/controlLayers/components/BrushSize';
import ControlLayersSettingsPopover from 'features/controlLayers/components/ControlLayersSettingsPopover';
import { ToolChooser } from 'features/controlLayers/components/ToolChooser';
import { UndoRedoButtonGroup } from 'features/controlLayers/components/UndoRedoButtonGroup';
import { memo } from 'react';

export const ControlLayersToolbar = memo(() => {
  return (
    <Flex gap={4}>
      <BrushSize />
      <ToolChooser />
      <UndoRedoButtonGroup />
      <ControlLayersSettingsPopover />
    </Flex>
  );
});

ControlLayersToolbar.displayName = 'ControlLayersToolbar';
