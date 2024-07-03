import { Spacer } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { CanvasEntityDeleteButton } from 'features/controlLayers/components/common/CanvasEntityDeleteButton';
import { CanvasEntityEnabledToggle } from 'features/controlLayers/components/common/CanvasEntityEnabledToggle';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityTitle } from 'features/controlLayers/components/common/CanvasEntityTitle';
import { LayerActionsMenu } from 'features/controlLayers/components/Layer/LayerActionsMenu';
import { layerDeleted, layerIsEnabledToggled } from 'features/controlLayers/store/canvasV2Slice';
import { selectLayerOrThrow } from 'features/controlLayers/store/layersReducers';
import { memo, useCallback, useMemo } from 'react';
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
  const objectCount = useAppSelector((s) => selectLayerOrThrow(s.canvasV2, id).objects.length);
  const onToggleIsEnabled = useCallback(() => {
    dispatch(layerIsEnabledToggled({ id }));
  }, [dispatch, id]);
  const onDelete = useCallback(() => {
    dispatch(layerDeleted({ id }));
  }, [dispatch, id]);
  const title = useMemo(() => {
    return `${t('controlLayers.layer')} (${t('controlLayers.objects', { count: objectCount })})`;
  }, [objectCount, t]);

  return (
    <CanvasEntityHeader onToggle={onToggleVisibility}>
      <CanvasEntityEnabledToggle isEnabled={isEnabled} onToggle={onToggleIsEnabled} />
      <CanvasEntityTitle title={title} />
      <Spacer />
      <LayerOpacity id={id} />
      <LayerActionsMenu id={id} />
      <CanvasEntityDeleteButton onDelete={onDelete} />
    </CanvasEntityHeader>
  );
});

LayerHeader.displayName = 'LayerHeader';
