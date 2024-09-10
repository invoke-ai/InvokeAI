import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

type Props = {
  onDelete: () => void;
};

export const RegionalGuidanceDeletePromptButton = memo(({ onDelete }: Props) => {
  const { t } = useTranslation();
  return (
    <Tooltip label={t('controlLayers.deletePrompt')}>
      <IconButton
        variant="link"
        aria-label={t('controlLayers.deletePrompt')}
        icon={<PiTrashSimpleBold />}
        onClick={onDelete}
        flexGrow={0}
        size="sm"
        p={0}
      />
    </Tooltip>
  );
});

RegionalGuidanceDeletePromptButton.displayName = 'RegionalGuidanceDeletePromptButton';
