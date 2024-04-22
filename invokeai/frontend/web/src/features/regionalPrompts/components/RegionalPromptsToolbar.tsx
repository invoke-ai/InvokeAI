/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { BrushSize } from 'features/regionalPrompts/components/BrushSize';
import { GlobalMaskLayerOpacity } from 'features/regionalPrompts/components/GlobalMaskLayerOpacity';
import { ToolChooser } from 'features/regionalPrompts/components/ToolChooser';
import { UndoRedoButtonGroup } from 'features/regionalPrompts/components/UndoRedoButtonGroup';
import { memo } from 'react';

export const RegionalPromptsToolbar = memo(() => {
  return (
    <Flex gap={4}>
      <BrushSize />
      <GlobalMaskLayerOpacity />
      <UndoRedoButtonGroup />
      <ToolChooser />
    </Flex>
  );
});

RegionalPromptsToolbar.displayName = 'RegionalPromptsToolbar';
