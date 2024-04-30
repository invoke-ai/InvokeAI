import { Divider, Flex, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { guidanceLayerIPAdapterDeleted } from 'app/store/middleware/listenerMiddleware/listeners/regionalControlToControlAdapterBridge';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import ControlAdapterLayerConfig from 'features/regionalPrompts/components/controlAdapterOverrides/ControlAdapterLayerConfig';
import { isMaskedGuidanceLayer, selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import { PiTrashSimpleBold } from 'react-icons/pi';
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

  if (ipAdapterIds.length === 0) {
    return null;
  }

  return (
    <>
      {ipAdapterIds.map((id, index) => (
        <Flex flexDir="column" key={id}>
          {index > 0 && (
            <Flex pb={3}>
              <Divider />
            </Flex>
          )}
          <IPAdapterListItem layerId={layerId} ipAdapterId={id} ipAdapterNumber={index + 1} />
        </Flex>
      ))}
    </>
  );
});

RPLayerIPAdapterList.displayName = 'RPLayerIPAdapterList';

type IPAdapterListItemProps = {
  layerId: string;
  ipAdapterId: string;
  ipAdapterNumber: number;
};

const IPAdapterListItem = memo(({ layerId, ipAdapterId, ipAdapterNumber }: IPAdapterListItemProps) => {
  const dispatch = useAppDispatch();
  const onDeleteIPAdapter = useCallback(() => {
    dispatch(guidanceLayerIPAdapterDeleted({ layerId, ipAdapterId }));
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
      <ControlAdapterLayerConfig id={ipAdapterId} />
    </Flex>
  );
});

IPAdapterListItem.displayName = 'IPAdapterListItem';
