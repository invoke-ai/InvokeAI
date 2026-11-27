import { IconButton, Tooltip } from '@invoke-ai/ui-library';
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
    <Tooltip label={t('prompt.addPromptTrigger')}>
      <IconButton
        variant="promptOverlay"
        isDisabled={isOpen}
        aria-label={t('prompt.addPromptTrigger')}
        icon={<PiCodeBold />}
        onClick={onOpen}
      />
    </Tooltip>
  );
});

AddPromptTriggerButton.displayName = 'AddPromptTriggerButton';
