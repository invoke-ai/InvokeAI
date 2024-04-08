import { Flex } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { AddLayerButton } from 'features/regionalPrompts/components/AddLayerButton';
import { BrushSize } from 'features/regionalPrompts/components/BrushSize';
import { LayerListItem } from 'features/regionalPrompts/components/LayerListItem';
import { RegionalPromptsStage } from 'features/regionalPrompts/components/RegionalPromptsStage';
import { selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';

const selectLayerIds = createSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
  regionalPrompts.layers.map((l) => l.id)
);

export const RegionalPromptsEditor = () => {
  const layerIds = useAppSelector(selectLayerIds);
  return (
    <Flex gap={4}>
      <Flex flexDir="column" w={200}>
        <AddLayerButton />
        <BrushSize />
        {layerIds.map((id) => (
          <LayerListItem key={id} id={id} />
        ))}
      </Flex>
      <Flex>
        <RegionalPromptsStage />
      </Flex>
    </Flex>
  );
};
