import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCodeBold } from 'react-icons/pi';

type Props = {
  isOpen: boolean;
  onOpen: () => void;
  isDisabled?: boolean;
};

export const AddPromptTriggerButton = memo((props: Props) => {
  const { onOpen, isOpen, isDisabled = false } = props;
  const { t } = useTranslation();
  
  const handleClick = useCallback(() => {
    if (!isDisabled) {
      onOpen();
    }
  }, [onOpen, isDisabled]);
  
  return (
    <Tooltip label={t('prompt.addPromptTrigger')}>
      <IconButton
        variant="promptOverlay"
        isDisabled={isOpen || isDisabled}
        aria-label={t('prompt.addPromptTrigger')}
        icon={<PiCodeBold />}
        onClick={handleClick}
      />
    </Tooltip>
  );
});

AddPromptTriggerButton.displayName = 'AddPromptTriggerButton';
