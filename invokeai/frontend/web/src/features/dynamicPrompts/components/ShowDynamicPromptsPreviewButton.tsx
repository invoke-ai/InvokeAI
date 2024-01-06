import type { SystemStyleObject } from '@chakra-ui/styled-system';
import { useAppSelector } from 'app/store/storeHooks';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { useDynamicPromptsModal } from 'features/dynamicPrompts/hooks/useDynamicPromptsModal';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { BsBracesAsterisk } from 'react-icons/bs';
import { spinAnimation } from 'theme/animations';

const loadingStyles: SystemStyleObject = {
  svg: { animation: spinAnimation },
};

export const ShowDynamicPromptsPreviewButton = memo(() => {
  const { t } = useTranslation();
  const isLoading = useAppSelector((state) => state.dynamicPrompts.isLoading);
  const { isOpen, onOpen } = useDynamicPromptsModal();
  return (
    <InvTooltip
      label={
        isLoading
          ? t('dynamicPrompts.loading')
          : t('dynamicPrompts.showDynamicPrompts')
      }
    >
      <InvIconButton
        size="sm"
        variant="promptOverlay"
        isDisabled={isOpen}
        aria-label={t('dynamicPrompts.showDynamicPrompts')}
        icon={<BsBracesAsterisk />}
        onClick={onOpen}
        sx={isLoading ? loadingStyles : undefined}
      />
    </InvTooltip>
  );
});

ShowDynamicPromptsPreviewButton.displayName = 'ShowDynamicPromptsPreviewButton';
