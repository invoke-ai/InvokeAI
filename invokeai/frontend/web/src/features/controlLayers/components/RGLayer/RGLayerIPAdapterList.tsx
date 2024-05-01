import { Divider, Flex, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  isRegionalGuidanceLayer,
  rgLayerIPAdapterDeleted,
  selectControlLayersSlice,
} from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback, useMemo } from 'react';
import { PiTrashSimpleBold } from 'react-icons/pi';
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
          <RGLayerIPAdapterListItem layerId={layerId} ipAdapterId={id} ipAdapterNumber={index + 1} />
        </Flex>
      ))}
    </>
  );
});

RGLayerIPAdapterList.displayName = 'RGLayerIPAdapterList';

type IPAdapterListItemProps = {
  layerId: string;
  ipAdapterId: string;
  ipAdapterNumber: number;
};

const RGLayerIPAdapterListItem = memo(({ layerId, ipAdapterId, ipAdapterNumber }: IPAdapterListItemProps) => {
  const dispatch = useAppDispatch();
  const onDeleteIPAdapter = useCallback(() => {
    dispatch(rgLayerIPAdapterDeleted({ layerId, ipAdapterId }));
  }, [dispatch, ipAdapterId, layerId]);

  return (
    <Flex flexDir="column" gap={3}>
      <Flex alignItems="center" gap={3}>
        <Text fontWeight="semibold" color="base.400">{`IP Adapter ${ipAdapterNumber}`}</Text>
        <Spacer />
        <IconButton
          size="sm"
          icon={<PiTrashSimpleBold />}
          aria-label="Delete IP Adapter"
          onClick={onDeleteIPAdapter}
          variant="ghost"
          colorScheme="error"
        />
      </Flex>
      {/* <ControlAdapterLayerConfig id={ipAdapterId} /> */}
    </Flex>
  );
});

RGLayerIPAdapterListItem.displayName = 'RGLayerIPAdapterListItem';
