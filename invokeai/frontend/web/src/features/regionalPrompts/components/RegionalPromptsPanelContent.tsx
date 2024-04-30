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
import { isRenderableLayer, selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import type { Layer } from 'features/regionalPrompts/store/types';
import { partition } from 'lodash-es';
import { memo } from 'react';

const selectLayerIdTypePairs = createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
  const [renderableLayers, ipAdapterLayers] = partition(regionalPrompts.present.layers, isRenderableLayer);
  return [...ipAdapterLayers, ...renderableLayers].map((l) => ({ id: l.id, type: l.type })).reverse();
});

export const RegionalPromptsPanelContent = memo(() => {
  const layerIdTypePairs = useAppSelector(selectLayerIdTypePairs);
  return (
    <Flex flexDir="column" gap={4} w="full" h="full">
      <Flex justifyContent="space-around">
        <AddLayerButton />
        <DeleteAllLayersButton />
      </Flex>
      <ScrollableContent>
        <Flex flexDir="column" gap={4}>
          {layerIdTypePairs.map(({ id, type }) => (
            <LayerWrapper key={id} id={id} type={type} />
          ))}
        </Flex>
      </ScrollableContent>
    </Flex>
  );
});

RegionalPromptsPanelContent.displayName = 'RegionalPromptsPanelContent';

type LayerWrapperProps = {
  id: string;
  type: Layer['type'];
};

const LayerWrapper = memo(({ id, type }: LayerWrapperProps) => {
  if (type === 'masked_guidance_layer') {
    return <MaskedGuidanceLayerListItem key={id} layerId={id} />;
  }
  if (type === 'control_adapter_layer') {
    return <ControlAdapterLayerListItem key={id} layerId={id} />;
  }
  if (type === 'ip_adapter_layer') {
    return <IPAdapterLayerListItem key={id} layerId={id} />;
  }
});

LayerWrapper.displayName = 'LayerWrapper';
