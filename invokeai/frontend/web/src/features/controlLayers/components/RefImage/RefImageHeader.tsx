import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, IconButton, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useRefImageEntity } from 'features/controlLayers/components/RefImage/useRefImageEntity';
import { useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { refImageDeleted, selectRefImageEntityIds } from 'features/controlLayers/store/refImagesSlice';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiGearBold, PiTrashBold } from 'react-icons/pi';

const textSx: SystemStyleObject = {
  color: 'base.300',
  '&[data-is-error="true"]': {
    color: 'error.300',
  },
};

export const RefImageHeader = memo(() => {
  const dispatch = useAppDispatch();
  const id = useRefImageIdContext();
  const { t } = useTranslation();
  const selectRefImageNumber = useMemo(
    () => createSelector(selectRefImageEntityIds, (ids) => ids.indexOf(id) + 1),
    [id]
  );
  const refImageNumber = useAppSelector(selectRefImageNumber);
  const entity = useRefImageEntity(id);
  const deleteRefImage = useCallback(() => {
    dispatch(refImageDeleted({ id }));
  }, [dispatch, id]);

  // Advanced section toggle for IP Adapter settings
  const { isOpen: isAdvancedOpen, onToggle: onToggleAdvanced } = useStandaloneAccordionToggle({
    id: `reference-image-advanced-${id}`,
    defaultIsOpen: false,
  });

  return (
    <Flex justifyContent="space-between" alignItems="center" w="full" ps={2}>
      <Text fontWeight="semibold" sx={textSx} data-is-error={!entity.config.image}>
        Reference Image #{refImageNumber}
      </Text>
      <Flex gap={1}>
        <IconButton
          tooltip={t('accordions.advanced.title')}
          size="xs"
          variant="link"
          alignSelf="stretch"
          aria-label={t('accordions.advanced.title')}
          onClick={onToggleAdvanced}
          icon={<PiGearBold />}
          colorScheme={isAdvancedOpen ? 'accent' : 'base'}
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
