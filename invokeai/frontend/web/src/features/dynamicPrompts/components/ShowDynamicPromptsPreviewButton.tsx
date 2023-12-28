import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { useDynamicPromptsModal } from 'features/dynamicPrompts/hooks/useDynamicPromptsModal';
import { useTranslation } from 'react-i18next';
import { BsBracesAsterisk } from 'react-icons/bs';

export const ShowDynamicPromptsPreviewButton = () => {
  const { t } = useTranslation();
  const { isOpen, onOpen } = useDynamicPromptsModal();
  return (
    <InvTooltip label={t('dynamicPrompts.showDynamicPrompts')}>
      <InvIconButton
        size="sm"
        variant="promptOverlay"
        isDisabled={isOpen}
        aria-label={t('dynamicPrompts.showDynamicPrompts')}
        icon={<BsBracesAsterisk />}
        onClick={onOpen}
      />
    </InvTooltip>
  );
};
