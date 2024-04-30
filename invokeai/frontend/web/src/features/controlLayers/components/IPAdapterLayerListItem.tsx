import { Flex, Spacer } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import ControlAdapterLayerConfig from 'features/controlLayers/components/controlAdapterOverrides/ControlAdapterLayerConfig';
import { LayerTitle } from 'features/controlLayers/components/LayerTitle';
import { RPLayerDeleteButton } from 'features/controlLayers/components/RPLayerDeleteButton';
import { RPLayerVisibilityToggle } from 'features/controlLayers/components/RPLayerVisibilityToggle';
import { isIPAdapterLayer, selectRegionalPromptsSlice } from 'features/controlLayers/store/regionalPromptsSlice';
import { memo, useMemo } from 'react';
import { assert } from 'tsafe';

type Props = {
  layerId: string;
};

export const IPAdapterLayerListItem = memo(({ layerId }: Props) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === layerId);
        assert(isIPAdapterLayer(layer), `Layer ${layerId} not found or not an IP Adapter layer`);
        return layer.ipAdapterId;
      }),
    [layerId]
  );
  const ipAdapterId = useAppSelector(selector);
  return (
    <Flex gap={2} bg="base.800" borderRadius="base" p="1px">
      <Flex flexDir="column" gap={4} w="full" bg="base.850" p={3} borderRadius="base">
        <Flex gap={3} alignItems="center">
          <RPLayerVisibilityToggle layerId={layerId} />
          <LayerTitle type="ip_adapter_layer" />
          <Spacer />
          <RPLayerDeleteButton layerId={layerId} />
        </Flex>
        <ControlAdapterLayerConfig id={ipAdapterId} />
      </Flex>
    </Flex>
  );
});

IPAdapterLayerListItem.displayName = 'IPAdapterLayerListItem';
