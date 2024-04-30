import { Flex, Spacer, useDisclosure } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import ControlAdapterLayerConfig from 'features/controlLayers/components/controlAdapterOverrides/ControlAdapterLayerConfig';
import { LayerDeleteButton } from 'features/controlLayers/components/LayerDeleteButton';
import { LayerTitle } from 'features/controlLayers/components/LayerTitle';
import { LayerVisibilityToggle } from 'features/controlLayers/components/LayerVisibilityToggle';
import { isIPAdapterLayer, selectControlLayersSlice } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useMemo } from 'react';
import { assert } from 'tsafe';

type Props = {
  layerId: string;
};

export const IPLayerListItem = memo(({ layerId }: Props) => {
  const selector = useMemo(
    () =>
      createMemoizedSelector(selectControlLayersSlice, (controlLayers) => {
        const layer = controlLayers.present.layers.find((l) => l.id === layerId);
        assert(isIPAdapterLayer(layer), `Layer ${layerId} not found or not an IP Adapter layer`);
        return layer.ipAdapterId;
      }),
    [layerId]
  );
  const ipAdapterId = useAppSelector(selector);
  const { isOpen, onToggle } = useDisclosure();
  return (
    <Flex gap={2} bg="base.800" borderRadius="base" p="1px" px={2}>
      <Flex flexDir="column" w="full" bg="base.850" borderRadius="base">
        <Flex gap={3} alignItems="center" p={3} cursor="pointer" onDoubleClick={onToggle}>
          <LayerVisibilityToggle layerId={layerId} />
          <LayerTitle type="ip_adapter_layer" />
          <Spacer />
          <LayerDeleteButton layerId={layerId} />
        </Flex>
        {isOpen && (
          <Flex gap={3} alignItems="center" px={3} pb={3} cursor="pointer">
            <ControlAdapterLayerConfig id={ipAdapterId} />
          </Flex>
        )}
      </Flex>
    </Flex>
  );
});

IPLayerListItem.displayName = 'IPLayerListItem';
