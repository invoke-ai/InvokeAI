import { ButtonGroup, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { layerDeleted, layerReset } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiArrowCounterClockwiseBold, PiTrashSimpleBold } from 'react-icons/pi';

type Props = { layerId: string };

export const RPLayerActionsButtonGroup = memo(({ layerId }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const deleteLayer = useCallback(() => {
    dispatch(layerDeleted(layerId));
  }, [dispatch, layerId]);
  const resetLayer = useCallback(() => {
    dispatch(layerReset(layerId));
  }, [dispatch, layerId]);
  return (
    <ButtonGroup isAttached={false}>
      <IconButton
        size="sm"
        aria-label={t('regionalPrompts.resetRegion')}
        tooltip={t('regionalPrompts.resetRegion')}
        icon={<PiArrowCounterClockwiseBold />}
        onClick={resetLayer}
      />
      <IconButton
        size="sm"
        colorScheme="error"
        aria-label={t('common.delete')}
        tooltip={t('common.delete')}
        icon={<PiTrashSimpleBold />}
        onClick={deleteLayer}
      />
    </ButtonGroup>
  );
});

RPLayerActionsButtonGroup.displayName = 'RPLayerActionsButtonGroup';
