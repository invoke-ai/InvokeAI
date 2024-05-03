/* eslint-disable i18next/no-literal-string */
import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { AddLayerButton } from 'features/controlLayers/components/AddLayerButton';
import { CALayer } from 'features/controlLayers/components/CALayer/CALayer';
import { DeleteAllLayersButton } from 'features/controlLayers/components/DeleteAllLayersButton';
import { IPALayer } from 'features/controlLayers/components/IPALayer/IPALayer';
import { RGLayer } from 'features/controlLayers/components/RGLayer/RGLayer';
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
    return <RGLayer key={id} layerId={id} />;
  }
  if (type === 'control_adapter_layer') {
    return <CALayer key={id} layerId={id} />;
  }
  if (type === 'ip_adapter_layer') {
    return <IPALayer key={id} layerId={id} />;
  }
});

LayerWrapper.displayName = 'LayerWrapper';
