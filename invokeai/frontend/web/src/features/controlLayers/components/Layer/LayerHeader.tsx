import { Spacer } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { LayerActionsMenu } from 'features/controlLayers/components/Layer/LayerActionsMenu';
import { layerDeleted, layerIsEnabledToggled } from 'features/controlLayers/store/canvasV2Slice';
import { selectLayerOrThrow } from 'features/controlLayers/store/layersReducers';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import { LayerOpacity } from './LayerOpacity';

type Props = {
  id: string;
  onToggleVisibility: () => void;
};

export const LayerHeader = memo(({ id, onToggleVisibility }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const isEnabled = useAppSelector((s) => selectLayerOrThrow(s.canvasV2, id).isEnabled);
  const onToggleIsEnabled = useCallback(() => {
    dispatch(layerIsEnabledToggled({ id }));
  }, [dispatch, id]);
  const onDelete = useCallback(() => {
    dispatch(layerDeleted({ id }));
  }, [dispatch, id]);

  return (
    <CanvasEntityHeader onToggle={onToggleVisibility}>
      <CanvasEntityEnabledToggle isEnabled={isEnabled} onToggle={onToggleIsEnabled} />
      <CanvasEntityTitle title={t('controlLayers.layer')} />
      <Spacer />
      <LayerOpacity id={id} />
      <LayerActionsMenu id={id} />
      <CanvasEntityDeleteButton onDelete={onDelete} />
    </CanvasEntityHeader>
  );
});

LayerHeader.displayName = 'LayerHeader';
