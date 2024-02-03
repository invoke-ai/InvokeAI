import { IconButton, Tooltip } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCodeBold } from 'react-icons/pi';

type Props = {
  isOpen: boolean;
  onOpen: () => void;
};

export const AddEmbeddingButton = memo((props: Props) => {
  const { onOpen, isOpen } = props;
  const { t } = useTranslation();
  return (
    <Tooltip label={t('embedding.addEmbedding')}>
      <IconButton
        variant="promptOverlay"
        isDisabled={isOpen}
        aria-label={t('embedding.addEmbedding')}
        icon={<PiCodeBold />}
        onClick={onOpen}
      />
    </Tooltip>
  );
});

AddEmbeddingButton.displayName = 'AddEmbeddingButton';
