import { IconButton } from '@invoke-ai/ui-library';
import { IAITooltip } from 'common/components/IAITooltip';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCodeBold } from 'react-icons/pi';

type Props = {
  isOpen: boolean;
  onOpen: () => void;
};

export const AddPromptTriggerButton = memo((props: Props) => {
  const { onOpen, isOpen } = props;
  const { t } = useTranslation();
  return (
    <IAITooltip label={t('prompt.addPromptTrigger')}>
      <IconButton
        variant="promptOverlay"
        isDisabled={isOpen}
        aria-label={t('prompt.addPromptTrigger')}
        icon={<PiCodeBold />}
        onClick={onOpen}
      />
    </IAITooltip>
  );
});

AddPromptTriggerButton.displayName = 'AddPromptTriggerButton';
