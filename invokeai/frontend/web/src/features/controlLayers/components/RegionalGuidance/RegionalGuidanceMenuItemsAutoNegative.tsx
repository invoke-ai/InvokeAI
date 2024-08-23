import { MenuItem } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { rgAutoNegativeToggled } from 'features/controlLayers/store/canvasV2Slice';
import { selectRegionalGuidanceEntityOrThrow } from 'features/controlLayers/store/regionsReducers';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiSelectionInverseBold } from 'react-icons/pi';

export const RegionalGuidanceMenuItemsAutoNegative = memo(() => {
  const { id } = useEntityIdentifierContext();
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const autoNegative = useAppSelector((s) => selectRegionalGuidanceEntityOrThrow(s.canvasV2, id).autoNegative);
  const onClick = useCallback(() => {
    dispatch(rgAutoNegativeToggled({ id }));
  }, [dispatch, id]);

  return (
    <MenuItem icon={<PiSelectionInverseBold />} onClick={onClick}>
      {autoNegative ? t('controlLayers.disableAutoNegative') : t('controlLayers.enableAutoNegative')}
    </MenuItem>
  );
});

RegionalGuidanceMenuItemsAutoNegative.displayName = 'RegionalGuidanceMenuItemsAutoNegative';
