import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import { InvTooltip } from 'common/components/InvTooltip/InvTooltip';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaCode } from 'react-icons/fa';

type Props = {
  isOpen: boolean;
  onOpen: () => void;
};

export const AddEmbeddingButton = memo((props: Props) => {
  const { onOpen, isOpen } = props;
  const { t } = useTranslation();
  return (
    <InvTooltip label={t('embedding.addEmbedding')}>
      <InvIconButton
        variant="promptOverlay"
        isDisabled={isOpen}
        aria-label={t('embedding.addEmbedding')}
        icon={<FaCode />}
        onClick={onOpen}
      />
    </InvTooltip>
  );
});

AddEmbeddingButton.displayName = 'AddEmbeddingButton';
