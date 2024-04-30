import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import {
  maskLayerNegativePromptChanged,
  maskLayerPositivePromptChanged,
} from 'features/controlLayers/store/regionalPromptsSlice';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

type Props = {
  layerId: string;
  polarity: 'positive' | 'negative';
};

export const RPLayerPromptDeleteButton = memo(({ layerId, polarity }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onClick = useCallback(() => {
    if (polarity === 'positive') {
      dispatch(maskLayerPositivePromptChanged({ layerId, prompt: null }));
    } else {
      dispatch(maskLayerNegativePromptChanged({ layerId, prompt: null }));
    }
  }, [dispatch, layerId, polarity]);
  return (
    <Tooltip label={t('regionalPrompts.deletePrompt')}>
      <IconButton
        variant="promptOverlay"
        aria-label={t('regionalPrompts.deletePrompt')}
        icon={<PiTrashSimpleBold />}
        onClick={onClick}
      />
    </Tooltip>
  );
});

RPLayerPromptDeleteButton.displayName = 'RPLayerPromptDeleteButton';
