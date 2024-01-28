import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setShouldConcatSDXLStylePrompt } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiLinkSimpleBold, PiLinkSimpleBreakBold } from 'react-icons/pi';

export const SDXLConcatButton = memo(() => {
  const shouldConcatSDXLStylePrompt = useAppSelector((s) => s.sdxl.shouldConcatSDXLStylePrompt);

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleShouldConcatPromptChange = useCallback(() => {
    dispatch(setShouldConcatSDXLStylePrompt(!shouldConcatSDXLStylePrompt));
  }, [dispatch, shouldConcatSDXLStylePrompt]);

  const label = useMemo(
    () => (shouldConcatSDXLStylePrompt ? t('sdxl.concatPromptStyle') : t('sdxl.freePromptStyle')),
    [shouldConcatSDXLStylePrompt, t]
  );

  return (
    <Tooltip label={label}>
      <IconButton
        aria-label={label}
        onClick={handleShouldConcatPromptChange}
        icon={shouldConcatSDXLStylePrompt ? <PiLinkSimpleBold size={14} /> : <PiLinkSimpleBreakBold size={14} />}
        variant="promptOverlay"
        fontSize={12}
        px={0.5}
      />
    </Tooltip>
  );
});

SDXLConcatButton.displayName = 'SDXLConcatButton';
