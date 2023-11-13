import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { FaLink } from 'react-icons/fa';
import { setShouldConcatSDXLStylePrompt } from '../store/sdxlSlice';
import { useTranslation } from 'react-i18next';
import { useCallback } from 'react';

export default function ParamSDXLConcatButton() {
  const shouldConcatSDXLStylePrompt = useAppSelector(
    (state: RootState) => state.sdxl.shouldConcatSDXLStylePrompt
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleShouldConcatPromptChange = useCallback(() => {
    dispatch(setShouldConcatSDXLStylePrompt(!shouldConcatSDXLStylePrompt));
  }, [dispatch, shouldConcatSDXLStylePrompt]);

  return (
    <IAIIconButton
      aria-label={t('sdxl.concatPromptStyle')}
      tooltip={t('sdxl.concatPromptStyle')}
      variant="outline"
      isChecked={shouldConcatSDXLStylePrompt}
      onClick={handleShouldConcatPromptChange}
      icon={<FaLink />}
      size="xs"
      sx={{
        position: 'absolute',
        insetInlineEnd: 1,
        top: 6,
        border: 'none',
        color: shouldConcatSDXLStylePrompt ? 'accent.500' : 'base.500',
        _hover: {
          bg: 'none',
        },
      }}
    ></IAIIconButton>
  );
}
