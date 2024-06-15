import { Spacer } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { CAActionsMenu } from 'features/controlLayers/components/ControlAdapter/CAActionsMenu';
import { CAOpacityAndFilter } from 'features/controlLayers/components/ControlAdapter/CAOpacityAndFilter';
import { caDeleted, caIsEnabledToggled } from 'features/controlLayers/store/canvasV2Slice';
import { selectCAOrThrow } from 'features/controlLayers/store/controlAdaptersReducers';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
  onToggleVisibility: () => void;
};

export const CAHeader = memo(({ id, onToggleVisibility }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isEnabled = useAppSelector((s) => selectCAOrThrow(s.canvasV2, id).isEnabled);
  const onToggleIsEnabled = useCallback(() => {
    dispatch(caIsEnabledToggled({ id }));
  }, [dispatch, id]);
  const onDelete = useCallback(() => {
    dispatch(caDeleted({ id }));
  }, [dispatch, id]);

  return (
    <CanvasEntityHeader onToggle={onToggleVisibility}>
      <CanvasEntityEnabledToggle isEnabled={isEnabled} onToggle={onToggleIsEnabled} />
      <CanvasEntityTitle title={t('controlLayers.globalControlAdapter')} />
      <Spacer />
      <CAOpacityAndFilter id={id} />
      <CAActionsMenu id={id} />
      <CanvasEntityDeleteButton onDelete={onDelete} />
    </CanvasEntityHeader>
  );
});

CAHeader.displayName = 'CAEntityHeader';
