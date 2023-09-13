import IAIIconButton from 'common/components/IAIIconButton';
import { memo } from 'react';
import { FaCode } from 'react-icons/fa';
import { useTranslation } from 'react-i18next';

type Props = {
  onClick: () => void;
};

const AddEmbeddingButton = (props: Props) => {
  const { onClick } = props;
  const { t } = useTranslation();
  return (
    <IAIIconButton
      size="sm"
      aria-label={t('embedding.addEmbedding')}
      tooltip={t('embedding.addEmbedding')}
      icon={<FaCode />}
      sx={{
        p: 2,
        color: 'base.500',
        _hover: {
          color: 'base.600',
        },
        _active: {
          color: 'base.700',
        },
        _dark: {
          color: 'base.500',
          _hover: {
            color: 'base.400',
          },
          _active: {
            color: 'base.300',
          },
        },
      }}
      variant="link"
      onClick={onClick}
    />
  );
};

export default memo(AddEmbeddingButton);
