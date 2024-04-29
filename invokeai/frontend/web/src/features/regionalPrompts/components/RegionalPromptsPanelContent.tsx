/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { AddLayerButton } from 'features/regionalPrompts/components/AddLayerButton';
import { ControlAdapterLayerListItem } from 'features/regionalPrompts/components/ControlAdapterLayerListItem';
import { DeleteAllLayersButton } from 'features/regionalPrompts/components/DeleteAllLayersButton';
import { IPAdapterLayerListItem } from 'features/regionalPrompts/components/IPAdapterLayerListItem';
import { MaskedGuidanceLayerListItem } from 'features/regionalPrompts/components/MaskedGuidanceLayerListItem';
import {
  isControlAdapterLayer,
  isIPAdapterLayer,
  isMaskedGuidanceLayer,
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo } from 'react';

const selectMaskedGuidanceLayerIds = createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
  regionalPrompts.present.layers
    .filter(isMaskedGuidanceLayer)
    .map((l) => l.id)
    .reverse()
);

const selectControlNetLayerIds = createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
  regionalPrompts.present.layers
    .filter(isControlAdapterLayer)
    .map((l) => l.id)
    .reverse()
);

const selectIPAdapterLayerIds = createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
  regionalPrompts.present.layers
    .filter(isIPAdapterLayer)
    .map((l) => l.id)
    .reverse()
);

export const RegionalPromptsPanelContent = memo(() => {
  const maskedGuidanceLayerIds = useAppSelector(selectMaskedGuidanceLayerIds);
  const controlNetLayerIds = useAppSelector(selectControlNetLayerIds);
  const ipAdapterLayerIds = useAppSelector(selectIPAdapterLayerIds);
  return (
    <Flex flexDir="column" gap={4} w="full" h="full">
      <Flex justifyContent="space-around">
        <AddLayerButton />
        <DeleteAllLayersButton />
      </Flex>
      <ScrollableContent>
        <Flex flexDir="column" gap={4}>
          {maskedGuidanceLayerIds.map((id) => (
            <MaskedGuidanceLayerListItem key={id} layerId={id} />
          ))}
          {controlNetLayerIds.map((id) => (
            <ControlAdapterLayerListItem key={id} layerId={id} />
          ))}
          {ipAdapterLayerIds.map((id) => (
            <IPAdapterLayerListItem key={id} layerId={id} />
          ))}
        </Flex>
      </ScrollableContent>
    </Flex>
  );
});

RegionalPromptsPanelContent.displayName = 'RegionalPromptsPanelContent';
