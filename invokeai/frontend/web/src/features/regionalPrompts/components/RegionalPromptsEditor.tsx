/* eslint-disable i18next/no-literal-string */
import { Button, ButtonGroup, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { AddLayerButton } from 'features/regionalPrompts/components/AddLayerButton';
import { BrushSize } from 'features/regionalPrompts/components/BrushSize';
import { DeleteAllLayersButton } from 'features/regionalPrompts/components/DeleteAllLayersButton';
import { LayerListItem } from 'features/regionalPrompts/components/LayerListItem';
import AutoNegativeCombobox from 'features/regionalPrompts/components/NegativeModeCombobox';
import { PromptLayerOpacity } from 'features/regionalPrompts/components/PromptLayerOpacity';
import { StageComponent } from 'features/regionalPrompts/components/StageComponent';
import { ToolChooser } from 'features/regionalPrompts/components/ToolChooser';
import { selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { getRegionalPromptLayerBlobs } from 'features/regionalPrompts/util/getLayerBlobs';
import { memo } from 'react';

const selectLayerIdsReversed = createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
  regionalPrompts.layers.map((l) => l.id).reverse()
);

const debugBlobs = () => {
  getRegionalPromptLayerBlobs(undefined, true);
};

export const RegionalPromptsEditor = memo(() => {
  const layerIdsReversed = useAppSelector(selectLayerIdsReversed);
  return (
    <Flex gap={4} w="full" h="full">
      <Flex flexDir="column" gap={4} flexShrink={0}>
        <Flex gap={3}>
          <ButtonGroup isAttached={false}>
            <Button onClick={debugBlobs}>DEBUG</Button>
            <AddLayerButton />
            <DeleteAllLayersButton />
          </ButtonGroup>
          <ToolChooser />
        </Flex>
        <BrushSize />
        <PromptLayerOpacity />
        <AutoNegativeCombobox />
        {layerIdsReversed.map((id) => (
          <LayerListItem key={id} id={id} />
        ))}
      </Flex>
      <StageComponent />
    </Flex>
  );
});

RegionalPromptsEditor.displayName = 'RegionalPromptsEditor';
