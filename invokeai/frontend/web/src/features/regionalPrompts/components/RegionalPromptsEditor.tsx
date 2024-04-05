import { Flex } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { RegionalPromptsStage } from 'features/regionalPrompts/components/RegionalPromptsStage';
import { layersSelectors, selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';

const selectLayers = createSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
  layersSelectors.selectAll(regionalPrompts)
);

export const RegionalPromptsEditor = () => {
  const layers = useAppSelector(selectLayers);
  return (
    <Flex>
      <Flex flexBasis={1}>
        {layers.map((layer) => (
          <Flex key={layer.id}>{layer.prompt}</Flex>
        ))}
      </Flex>
      <Flex flexBasis={1}>
        <RegionalPromptsStage />
      </Flex>
    </Flex>
  );
};
