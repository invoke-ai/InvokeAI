import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import { useEntityIsEnabled } from 'features/controlLayers/hooks/useEntityIsEnabled';
import { entityIsEnabledToggled } from 'features/controlLayers/store/canvasSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCircleBold, PiCircleFill } from 'react-icons/pi';

export const CanvasEntityEnabledToggle = memo(() => {
  const { t } = useTranslation();
  const entityIdentifier = useEntityIdentifierContext();
  const isEnabled = useEntityIsEnabled(entityIdentifier);
  const isBusy = useCanvasIsBusy();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    dispatch(entityIsEnabledToggled({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return (
    <IconButton
      size="sm"
      aria-label={t(isEnabled ? 'common.enabled' : 'common.disabled')}
      tooltip={t(isEnabled ? 'common.enabled' : 'common.disabled')}
      variant="link"
      alignSelf="stretch"
      icon={isEnabled ? <PiCircleFill /> : <PiCircleBold />}
      onClick={onClick}
      isDisabled={isBusy}
    />
  );
});

CanvasEntityEnabledToggle.displayName = 'CanvasEntityEnabledToggle';
