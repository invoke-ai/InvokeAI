import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiTrashSimpleBold } from 'react-icons/pi';

type Props = {
  onDelete: () => void;
};

export const RGDeletePromptButton = memo(({ onDelete }: Props) => {
  const { t } = useTranslation();
  return (
    <Tooltip label={t('controlLayers.deletePrompt')}>
      <IconButton
        variant="promptOverlay"
        aria-label={t('controlLayers.deletePrompt')}
        icon={<PiTrashSimpleBold />}
        onClick={onDelete}
      />
    </Tooltip>
  );
});

RGDeletePromptButton.displayName = 'RGDeletePromptButton';
