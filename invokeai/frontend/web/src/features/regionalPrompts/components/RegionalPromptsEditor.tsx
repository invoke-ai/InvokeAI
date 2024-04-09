import { Flex } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { AddLayerButton } from 'features/regionalPrompts/components/AddLayerButton';
import { BrushSize } from 'features/regionalPrompts/components/BrushSize';
import { LayerListItem } from 'features/regionalPrompts/components/LayerListItem';
import { RegionalPromptsStage } from 'features/regionalPrompts/components/RegionalPromptsStage';
import { selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';

const selectLayerIdsReversed = createSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
  regionalPrompts.layers.map((l) => l.id).reverse()
);

export const RegionalPromptsEditor = () => {
  const layerIdsReversed = useAppSelector(selectLayerIdsReversed);
  return (
    <Flex gap={4}>
      <Flex flexDir="column" w={200} gap={4}>
        <AddLayerButton />
        <BrushSize />
        {layerIdsReversed.map((id) => (
          <LayerListItem key={id} id={id} />
        ))}
      </Flex>
      <Flex>
        <RegionalPromptsStage />
      </Flex>
    </Flex>
  );
};
