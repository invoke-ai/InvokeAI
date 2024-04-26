/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { AddLayerButton } from 'features/regionalPrompts/components/AddLayerButton';
import { DeleteAllLayersButton } from 'features/regionalPrompts/components/DeleteAllLayersButton';
import { RPLayerListItem } from 'features/regionalPrompts/components/RPLayerListItem';
import { isMaskedGuidanceLayer, selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo } from 'react';

const selectRPLayerIdsReversed = createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
  regionalPrompts.present.layers
    .filter(isMaskedGuidanceLayer)
    .map((l) => l.id)
    .reverse()
);

export const RegionalPromptsPanelContent = memo(() => {
  const rpLayerIdsReversed = useAppSelector(selectRPLayerIdsReversed);
  return (
    <Flex flexDir="column" gap={4} w="full" h="full">
      <Flex justifyContent="space-around">
        <AddLayerButton />
        <DeleteAllLayersButton />
      </Flex>
      <ScrollableContent>
        <Flex flexDir="column" gap={4}>
          {rpLayerIdsReversed.map((id) => (
            <RPLayerListItem key={id} layerId={id} />
          ))}
        </Flex>
      </ScrollableContent>
    </Flex>
  );
});

RegionalPromptsPanelContent.displayName = 'RegionalPromptsPanelContent';
