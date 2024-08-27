import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { IconButton, spinAnimation, Tooltip } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { useDynamicPromptsModal } from 'features/dynamicPrompts/hooks/useDynamicPromptsModal';
import { selectDynamicPromptsSlice } from 'features/dynamicPrompts/store/dynamicPromptsSlice';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { BsBracesAsterisk } from 'react-icons/bs';

const loadingStyles: SystemStyleObject = {
  svg: { animation: spinAnimation },
};

const selectIsError = createSelector(selectDynamicPromptsSlice, (dynamicPrompts) =>
  Boolean(dynamicPrompts.isError || dynamicPrompts.parsingError)
);
const selectIsLoading = createSelector(selectDynamicPromptsSlice, (dynamicPrompts) => dynamicPrompts.isLoading);

export const ShowDynamicPromptsPreviewButton = memo(() => {
  const { t } = useTranslation();
  const isLoading = useAppSelector(selectIsLoading);
  const isError = useAppSelector(selectIsError);
  const { isOpen, onOpen } = useDynamicPromptsModal();

  return (
    <Tooltip label={isLoading ? t('dynamicPrompts.loading') : t('dynamicPrompts.showDynamicPrompts')}>
      <IconButton
        size="sm"
        variant="promptOverlay"
        isDisabled={isOpen}
        aria-label={t('dynamicPrompts.showDynamicPrompts')}
        icon={<BsBracesAsterisk />}
        onClick={onOpen}
        sx={isLoading ? loadingStyles : undefined}
        colorScheme={isError && !isLoading ? 'error' : 'base'}
      />
    </Tooltip>
  );
});

ShowDynamicPromptsPreviewButton.displayName = 'ShowDynamicPromptsPreviewButton';
