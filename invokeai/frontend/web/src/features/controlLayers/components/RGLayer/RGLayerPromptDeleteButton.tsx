import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import {
  regionalGuidanceNegativePromptChanged,
  regionalGuidancePositivePromptChanged,
} from 'features/controlLayers/store/controlLayersSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

type Props = {
  layerId: string;
  polarity: 'positive' | 'negative';
};

export const RGLayerPromptDeleteButton = memo(({ layerId, polarity }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    if (polarity === 'positive') {
      dispatch(regionalGuidancePositivePromptChanged({ layerId, prompt: null }));
    } else {
      dispatch(regionalGuidanceNegativePromptChanged({ layerId, prompt: null }));
    }
  }, [dispatch, layerId, polarity]);
  return (
    <Tooltip label={t('controlLayers.deletePrompt')}>
      <IconButton
        variant="promptOverlay"
        aria-label={t('controlLayers.deletePrompt')}
        icon={<PiTrashSimpleBold />}
        onClick={onClick}
      />
    </Tooltip>
  );
});

RGLayerPromptDeleteButton.displayName = 'RGLayerPromptDeleteButton';
