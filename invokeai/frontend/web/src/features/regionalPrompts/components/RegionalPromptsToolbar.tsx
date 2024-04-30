/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { BrushSize } from 'features/regionalPrompts/components/BrushSize';
import ControlLayersSettingsPopover from 'features/regionalPrompts/components/ControlLayersSettingsPopover';
import { ToolChooser } from 'features/regionalPrompts/components/ToolChooser';
import { UndoRedoButtonGroup } from 'features/regionalPrompts/components/UndoRedoButtonGroup';
import { memo } from 'react';

export const RegionalPromptsToolbar = memo(() => {
  return (
    <Flex gap={4}>
      <BrushSize />
      <ToolChooser />
      <UndoRedoButtonGroup />
      <ControlLayersSettingsPopover />
    </Flex>
  );
});

RegionalPromptsToolbar.displayName = 'RegionalPromptsToolbar';
