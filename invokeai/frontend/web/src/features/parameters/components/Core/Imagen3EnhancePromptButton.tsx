import { IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { imagen3EnhancePromptChanged, selectImagen3EnhancePrompt } from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiMagicWandFill } from 'react-icons/pi';

export const Imagen3EnhancePromptButton = memo(() => {
  const imagen3EnhancePrompt = useAppSelector(selectImagen3EnhancePrompt);

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const onClick = useCallback(() => {
    dispatch(imagen3EnhancePromptChanged(!imagen3EnhancePrompt));
  }, [dispatch, imagen3EnhancePrompt]);

  const label = useMemo(
    () => (imagen3EnhancePrompt ? t('imagen3.enhancePrompt') : t('imagen3.notEnhancePrompt')),
    [imagen3EnhancePrompt, t]
  );

  return (
    <IconButton
      tooltip={label}
      aria-label={label}
      onClick={onClick}
      icon={imagen3EnhancePrompt ? <PiMagicWandFill size={14} /> : <PiMagicWandFill size={14} />}
      colorScheme={imagen3EnhancePrompt ? 'green' : 'base'}
      variant="promptOverlay"
      fontSize={12}
      px={0.5}
    />
  );
});

Imagen3EnhancePromptButton.displayName = 'Imagen3EnhancePromptButton';
