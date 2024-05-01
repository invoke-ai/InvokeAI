import { Divider, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { RGLayerIPAdapterWrapper } from 'features/controlLayers/components/RGLayer/RGLayerIPAdapterWrapper';
import { isRegionalGuidanceLayer, selectControlLayersSlice } from 'features/controlLayers/store/controlLayersSlice';
import { memo, useMemo } from 'react';
import { assert } from 'tsafe';

type Props = {
  layerId: string;
};

export const RGLayerIPAdapterList = memo(({ layerId }: Props) => {
  const selectIPAdapterIds = useMemo(
    () =>
      createMemoizedSelector(selectControlLayersSlice, (controlLayers) => {
        const layer = controlLayers.present.layers.filter(isRegionalGuidanceLayer).find((l) => l.id === layerId);
        assert(layer, `Layer ${layerId} not found`);
        return layer.ipAdapters;
      }),
    [layerId]
  );
  const ipAdapters = useAppSelector(selectIPAdapterIds);

  if (ipAdapters.length === 0) {
    return null;
  }

  return (
    <>
      {ipAdapters.map(({ id }, index) => (
        <Flex flexDir="column" key={id}>
          {index > 0 && (
            <Flex pb={3}>
              <Divider />
            </Flex>
          )}
          <RGLayerIPAdapterWrapper layerId={layerId} ipAdapterId={id} ipAdapterNumber={index + 1} />
        </Flex>
      ))}
    </>
  );
});

RGLayerIPAdapterList.displayName = 'RGLayerIPAdapterList';
