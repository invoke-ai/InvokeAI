/* eslint-disable i18next/no-literal-string */
import { Button, ButtonGroup, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { AddLayerButton } from 'features/regionalPrompts/components/AddLayerButton';
import { BrushSize } from 'features/regionalPrompts/components/BrushSize';
import { DeleteAllLayersButton } from 'features/regionalPrompts/components/DeleteAllLayersButton';
import { PromptLayerOpacity } from 'features/regionalPrompts/components/PromptLayerOpacity';
import { RPEnabledSwitch } from 'features/regionalPrompts/components/RPEnabledSwitch';
import { RPLayerListItem } from 'features/regionalPrompts/components/RPLayerListItem';
import { StageComponent } from 'features/regionalPrompts/components/StageComponent';
import { ToolChooser } from 'features/regionalPrompts/components/ToolChooser';
import { UndoRedoButtonGroup } from 'features/regionalPrompts/components/UndoRedoButtonGroup';
import { isRPLayer, selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { getRegionalPromptLayerBlobs } from 'features/regionalPrompts/util/getLayerBlobs';
import { memo } from 'react';

const selectRPLayerIdsReversed = createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
  regionalPrompts.present.layers
    .filter(isRPLayer)
    .map((l) => l.id)
    .reverse()
);

const debugBlobs = () => {
  getRegionalPromptLayerBlobs(undefined, true);
};

export const RegionalPromptsEditor = memo(() => {
  const rpLayerIdsReversed = useAppSelector(selectRPLayerIdsReversed);
  return (
    <Flex gap={4} w="full" h="full">
      <Flex flexDir="column" gap={4} flexShrink={0} w={430}>
        <Flex gap={3}>
          <ButtonGroup isAttached={false}>
            <Button onClick={debugBlobs}>🐛</Button>
            <AddLayerButton />
            <DeleteAllLayersButton />
          </ButtonGroup>
          <UndoRedoButtonGroup />
          <ToolChooser />
        </Flex>
        <RPEnabledSwitch />
        <BrushSize />
        <PromptLayerOpacity />
        {rpLayerIdsReversed.map((id) => (
          <RPLayerListItem key={id} layerId={id} />
        ))}
      </Flex>
      <StageComponent />
    </Flex>
  );
});

RegionalPromptsEditor.displayName = 'RegionalPromptsEditor';
