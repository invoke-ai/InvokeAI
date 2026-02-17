import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { IconButton, spinAnimation } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IAITooltip } from 'common/components/IAITooltip';
import { useDynamicPromptsModal } from 'features/dynamicPrompts/hooks/useDynamicPromptsModal';
import {
  selectDynamicPromptsIsError,
  selectDynamicPromptsIsLoading,
} from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBracketsCurlyBold } from 'react-icons/pi';

const loadingStyles: SystemStyleObject = {
  svg: { animation: spinAnimation },
};

export const ShowDynamicPromptsPreviewButton = memo(() => {
  const { t } = useTranslation();
  const isLoading = useAppSelector(selectDynamicPromptsIsLoading);
  const isError = useAppSelector(selectDynamicPromptsIsError);
  const { isOpen, onOpen } = useDynamicPromptsModal();

  return (
    <IAITooltip label={isLoading ? t('dynamicPrompts.loading') : t('dynamicPrompts.showDynamicPrompts')}>
      <IconButton
        size="sm"
        variant="promptOverlay"
        isDisabled={isOpen}
        aria-label={t('dynamicPrompts.showDynamicPrompts')}
        icon={<PiBracketsCurlyBold />}
        onClick={onOpen}
        sx={isLoading ? loadingStyles : undefined}
        colorScheme={isError && !isLoading ? 'error' : 'base'}
      />
    </IAITooltip>
  );
});

ShowDynamicPromptsPreviewButton.displayName = 'ShowDynamicPromptsPreviewButton';
