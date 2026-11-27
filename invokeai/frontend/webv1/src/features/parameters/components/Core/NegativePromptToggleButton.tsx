import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { negativePromptChanged, selectHasNegativePrompt } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusMinusBold } from 'react-icons/pi';

export const NegativePromptToggleButton = memo(() => {
  const { t } = useTranslation();
  const hasNegativePrompt = useAppSelector(selectHasNegativePrompt);

  const dispatch = useAppDispatch();

  const onClick = useCallback(() => {
    if (hasNegativePrompt) {
      dispatch(negativePromptChanged(null));
    } else {
      dispatch(negativePromptChanged(''));
    }
  }, [dispatch, hasNegativePrompt]);

  const label = useMemo(
    () => (hasNegativePrompt ? t('common.removeNegativePrompt') : t('common.addNegativePrompt')),
    [hasNegativePrompt, t]
  );

  return (
    <Tooltip label={label}>
      <IconButton
        aria-label={label}
        onClick={onClick}
        icon={<PiPlusMinusBold size={14} />}
        variant="promptOverlay"
        fontSize={12}
        px={0.5}
        colorScheme={hasNegativePrompt ? 'invokeBlue' : 'base'}
      />
    </Tooltip>
  );
});

NegativePromptToggleButton.displayName = 'NegativePromptToggleButton';
