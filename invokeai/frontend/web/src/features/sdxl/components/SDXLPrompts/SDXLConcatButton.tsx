import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { setShouldConcatSDXLStylePrompt } from 'features/sdxl/store/sdxlSlice';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaLink, FaUnlink } from 'react-icons/fa';

export const SDXLConcatButton = memo(() => {
  const shouldConcatSDXLStylePrompt = useAppSelector(
    (state) => state.sdxl.shouldConcatSDXLStylePrompt
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleShouldConcatPromptChange = useCallback(() => {
    dispatch(setShouldConcatSDXLStylePrompt(!shouldConcatSDXLStylePrompt));
  }, [dispatch, shouldConcatSDXLStylePrompt]);

  const label = useMemo(
    () =>
      shouldConcatSDXLStylePrompt
        ? t('sdxl.concatPromptStyle')
        : t('sdxl.freePromptStyle'),
    [shouldConcatSDXLStylePrompt, t]
  );

  return (
    <InvTooltip label={label}>
      <InvIconButton
        aria-label={label}
        onClick={handleShouldConcatPromptChange}
        icon={shouldConcatSDXLStylePrompt ? <FaLink /> : <FaUnlink />}
        variant="promptOverlay"
        fontSize={12}
        px={0.5}
      />
    </InvTooltip>
  );
});

SDXLConcatButton.displayName = 'SDXLConcatButton';
