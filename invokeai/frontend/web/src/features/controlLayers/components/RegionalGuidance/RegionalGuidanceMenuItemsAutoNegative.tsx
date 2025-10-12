import { MenuItem } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rgAutoNegativeToggled } from 'features/controlLayers/store/canvasSlice';
import { selectActiveCanvas, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSelectionInverseBold } from 'react-icons/pi';

export const RegionalGuidanceMenuItemsAutoNegative = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const selectAutoNegative = useMemo(
    () =>
      createSelector(
        selectActiveCanvas,
        (canvas) => selectEntityOrThrow(canvas, entityIdentifier, 'RegionalGuidanceMenuItemsAutoNegative').autoNegative
      ),
    [entityIdentifier]
  );
  const autoNegative = useAppSelector(selectAutoNegative);
  const onClick = useCallback(() => {
    dispatch(rgAutoNegativeToggled({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <MenuItem icon={<PiSelectionInverseBold />} onClick={onClick}>
      {autoNegative ? t('controlLayers.disableAutoNegative') : t('controlLayers.enableAutoNegative')}
    </MenuItem>
  );
});

RegionalGuidanceMenuItemsAutoNegative.displayName = 'RegionalGuidanceMenuItemsAutoNegative';
