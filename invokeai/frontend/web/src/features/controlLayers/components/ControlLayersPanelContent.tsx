/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { AddLayerButton } from 'features/controlLayers/components/AddLayerButton';
import { CALayerListItem } from 'features/controlLayers/components/CALayerListItem';
import { DeleteAllLayersButton } from 'features/controlLayers/components/DeleteAllLayersButton';
import { IPLayerListItem } from 'features/controlLayers/components/IPLayerListItem';
import { RGLayerListItem } from 'features/controlLayers/components/RGLayerListItem';
import { isRenderableLayer, selectControlLayersSlice } from 'features/controlLayers/store/controlLayersSlice';
import type { Layer } from 'features/controlLayers/store/types';
import { partition } from 'lodash-es';
import { memo } from 'react';

const selectLayerIdTypePairs = createMemoizedSelector(selectControlLayersSlice, (controlLayers) => {
  const [renderableLayers, ipAdapterLayers] = partition(controlLayers.present.layers, isRenderableLayer);
  return [...ipAdapterLayers, ...renderableLayers].map((l) => ({ id: l.id, type: l.type })).reverse();
});

export const ControlLayersPanelContent = memo(() => {
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

ControlLayersPanelContent.displayName = 'ControlLayersPanelContent';

type LayerWrapperProps = {
  id: string;
  type: Layer['type'];
};

const LayerWrapper = memo(({ id, type }: LayerWrapperProps) => {
  if (type === 'regional_guidance_layer') {
    return <RGLayerListItem key={id} layerId={id} />;
  }
  if (type === 'control_adapter_layer') {
    return <CALayerListItem key={id} layerId={id} />;
  }
  if (type === 'ip_adapter_layer') {
    return <IPLayerListItem key={id} layerId={id} />;
  }
});

LayerWrapper.displayName = 'LayerWrapper';
