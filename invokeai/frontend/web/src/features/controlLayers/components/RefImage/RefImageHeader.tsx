import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IAITooltip } from 'common/components/IAITooltip';
import { useRefImageEntity } from 'features/controlLayers/components/RefImage/useRefImageEntity';
import { useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { selectMainModelConfig } from 'features/controlLayers/store/paramsSlice';
import {
  refImageDeleted,
  refImageIsEnabledToggled,
  selectRefImageEntityIds,
} from 'features/controlLayers/store/refImagesSlice';
import { getGlobalReferenceImageWarnings } from 'features/controlLayers/store/validators';
import { memo, useCallback, useMemo } from 'react';
import { PiCircleBold, PiCircleFill, PiTrashBold, PiWarningBold } from 'react-icons/pi';

import { RefImageWarningTooltipContent } from './RefImageWarningTooltipContent';

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
  const mainModelConfig = useAppSelector(selectMainModelConfig);

  const warnings = useMemo(() => {
    return getGlobalReferenceImageWarnings(entity, mainModelConfig);
  }, [entity, mainModelConfig]);

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
        {warnings.length > 0 && (
          <IAITooltip label={<RefImageWarningTooltipContent warnings={warnings} />}>
            <IconButton
              as="span"
              size="sm"
              variant="link"
              alignSelf="stretch"
              aria-label="warnings"
              icon={<PiWarningBold />}
              colorScheme="warning"
            />
          </IAITooltip>
        )}
        {!entity.isEnabled && (
          <Text fontSize="xs" fontStyle="italic" color="base.400">
            Disabled
          </Text>
        )}
        <IAITooltip label={entity.isEnabled ? 'Disable Reference Image' : 'Enable Reference Image'}>
          <IconButton
            size="xs"
            variant="link"
            alignSelf="stretch"
            aria-label={entity.isEnabled ? 'Disable ref image' : 'Enable ref image'}
            onClick={toggleIsEnabled}
            icon={entity.isEnabled ? <PiCircleFill /> : <PiCircleBold />}
          />
        </IAITooltip>
        <IAITooltip label="Delete Reference Image">
          <IconButton
            size="xs"
            variant="link"
            alignSelf="stretch"
            aria-label="Delete ref image"
            onClick={deleteRefImage}
            icon={<PiTrashBold />}
            colorScheme="error"
          />
        </IAITooltip>
      </Flex>
    </Flex>
  );
});
RefImageHeader.displayName = 'RefImageHeader';
