import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useRefImageEntity } from 'features/controlLayers/components/RefImage/useRefImageEntity';
import { useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import {
  refImageDeleted,
  refImageIsEnabledToggled,
  selectRefImageEntityIds,
} from 'features/controlLayers/store/refImagesSlice';
import { memo, useCallback, useMemo } from 'react';
import { PiCircleBold, PiCircleFill, PiTrashBold } from 'react-icons/pi';

const textSx: SystemStyleObject = {
  color: 'base.300',
  '&[data-is-error="true"]': {
    color: 'error.300',
  },
};

export const RefImageHeader = memo(() => {
  const dispatch = useAppDispatch();
  const id = useRefImageIdContext();
  const selectRefImageNumber = useMemo(
    () => createSelector(selectRefImageEntityIds, (ids) => ids.indexOf(id) + 1),
    [id]
  );
  const refImageNumber = useAppSelector(selectRefImageNumber);
  const entity = useRefImageEntity(id);
  const deleteRefImage = useCallback(() => {
    dispatch(refImageDeleted({ id }));
  }, [dispatch, id]);

  const toggleIsEnabled = useCallback(() => {
    dispatch(refImageIsEnabledToggled({ id }));
  }, [dispatch, id]);

  return (
    <Flex justifyContent="space-between" alignItems="center" w="full" ps={2}>
      <Text fontWeight="semibold" sx={textSx} data-is-error={!entity.config.image}>
        Reference Image #{refImageNumber}
      </Text>
      <Flex alignItems="center" gap={1}>
        {!entity.isEnabled && (
          <Text fontSize="xs" fontStyle="italic" color="base.400">
            Disabled
          </Text>
        )}
        <IconButton
          tooltip={entity.isEnabled ? 'Disable Reference Image' : 'Enable Reference Image'}
          size="xs"
          variant="link"
          alignSelf="stretch"
          aria-label={entity.isEnabled ? 'Disable ref image' : 'Enable ref image'}
          onClick={toggleIsEnabled}
          icon={entity.isEnabled ? <PiCircleFill /> : <PiCircleBold />}
        />
        <IconButton
          tooltip="Delete Reference Image"
          size="xs"
          variant="link"
          alignSelf="stretch"
          aria-label="Delete ref image"
          onClick={deleteRefImage}
          icon={<PiTrashBold />}
          colorScheme="error"
        />
      </Flex>
    </Flex>
  );
});
RefImageHeader.displayName = 'RefImageHeader';
