import { useAppDispatch } from 'app/store/storeHooks';
import { IconMenuItem } from 'common/components/IconMenuItem';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { entityDuplicated } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCopyFill } from 'react-icons/pi';

export const CanvasEntityMenuItemsDuplicate = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const isBusy = useCanvasIsBusy();

  const onClick = useCallback(() => {
    dispatch(entityDuplicated({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <IconMenuItem
      aria-label={t('controlLayers.duplicate')}
      tooltip={t('controlLayers.duplicate')}
      onClick={onClick}
      icon={<PiCopyFill />}
      isDisabled={isBusy}
    />
  );
});

CanvasEntityMenuItemsDuplicate.displayName = 'CanvasEntityMenuItemsDuplicate';
