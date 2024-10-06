import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { useEntityTypeIsHidden } from 'features/controlLayers/hooks/useEntityTypeIsHidden';
import { useEntityTypeString } from 'features/controlLayers/hooks/useEntityTypeString';
import { allEntitiesOfTypeIsHiddenToggled } from 'features/controlLayers/store/canvasSlice';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import type { MouseEventHandler } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiEyeBold, PiEyeClosedBold } from 'react-icons/pi';

type Props = {
  type: CanvasEntityIdentifier['type'];
};

export const CanvasEntityTypeIsHiddenToggle = memo(({ type }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isHidden = useEntityTypeIsHidden(type);
  const typeString = useEntityTypeString(type, true);
  const onClick = useCallback<MouseEventHandler>(
    (e) => {
      e.stopPropagation();
      dispatch(allEntitiesOfTypeIsHiddenToggled({ type }));
    },
    [dispatch, type]
  );

  return (
    <IconButton
      size="sm"
      aria-label={t(isHidden ? 'controlLayers.hidingType' : 'controlLayers.showingType', { type: typeString })}
      tooltip={t(isHidden ? 'controlLayers.hidingType' : 'controlLayers.showingType', { type: typeString })}
      variant="link"
      icon={isHidden ? <PiEyeClosedBold /> : <PiEyeBold />}
      onClick={onClick}
      alignSelf="stretch"
    />
  );
});

CanvasEntityTypeIsHiddenToggle.displayName = 'CanvasEntityTypeIsHiddenToggle';
