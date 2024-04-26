import { Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import ControlAdapterConfig from 'features/controlAdapters/components/ControlAdapterConfig';
import { isMaskedGuidanceLayer, selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useMemo } from 'react';
import { assert } from 'tsafe';

type Props = {
  layerId: string;
};

export const RPLayerIPAdapterList = memo(({ layerId }: Props) => {
  const selectIPAdapterIds = useMemo(
    () =>
      createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.filter(isMaskedGuidanceLayer).find((l) => l.id === layerId);
        assert(layer, `Layer ${layerId} not found`);
        return layer.ipAdapterIds;
      }),
    [layerId]
  );
  const ipAdapterIds = useAppSelector(selectIPAdapterIds);

  return (
    <Flex w="full" flexDir="column" gap={2}>
      {ipAdapterIds.map((id, index) => (
        <ControlAdapterConfig key={id} id={id} number={index + 1} />
      ))}
    </Flex>
  );
});

RPLayerIPAdapterList.displayName = 'RPLayerIPAdapterList';
