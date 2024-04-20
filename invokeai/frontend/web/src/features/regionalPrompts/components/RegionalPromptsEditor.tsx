/* eslint-disable i18next/no-literal-string */
import { Flex, Spacer } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { AddLayerButton } from 'features/regionalPrompts/components/AddLayerButton';
import { BrushSize } from 'features/regionalPrompts/components/BrushSize';
import { DebugLayersButton } from 'features/regionalPrompts/components/DebugLayersButton';
import { DeleteAllLayersButton } from 'features/regionalPrompts/components/DeleteAllLayersButton';
import { PromptLayerOpacity } from 'features/regionalPrompts/components/PromptLayerOpacity';
import { RPEnabledSwitch } from 'features/regionalPrompts/components/RPEnabledSwitch';
import { RPLayerListItem } from 'features/regionalPrompts/components/RPLayerListItem';
import { StageComponent } from 'features/regionalPrompts/components/StageComponent';
import { ToolChooser } from 'features/regionalPrompts/components/ToolChooser';
import { UndoRedoButtonGroup } from 'features/regionalPrompts/components/UndoRedoButtonGroup';
import { isRPLayer, selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo } from 'react';

const selectRPLayerIdsReversed = createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
  regionalPrompts.present.layers
    .filter(isRPLayer)
    .map((l) => l.id)
    .reverse()
);

export const RegionalPromptsEditor = memo(() => {
  const rpLayerIdsReversed = useAppSelector(selectRPLayerIdsReversed);
  return (
    <Flex gap={4} w="full" h="full">
      <Flex flexDir="column" gap={4} minW={430}>
        <Flex gap={3} w="full" justifyContent="space-between">
          <DebugLayersButton />
          <AddLayerButton />
          <DeleteAllLayersButton />
          <Spacer />
          <UndoRedoButtonGroup />
          <ToolChooser />
        </Flex>
        <RPEnabledSwitch />
        <BrushSize />
        <PromptLayerOpacity />
        <ScrollableContent>
          <Flex flexDir="column" gap={2}>
            {rpLayerIdsReversed.map((id) => (
              <RPLayerListItem key={id} layerId={id} />
            ))}
          </Flex>
        </ScrollableContent>
      </Flex>
      <StageComponent />
    </Flex>
  );
});

RegionalPromptsEditor.displayName = 'RegionalPromptsEditor';
