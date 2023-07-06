import IAIIconButton from 'common/components/IAIIconButton';
import { memo } from 'react';
import { BiCode } from 'react-icons/bi';

type Props = {
  onClick: () => void;
};

const AddEmbeddingButton = (props: Props) => {
  const { onClick } = props;
  return (
    <IAIIconButton
      size="sm"
      aria-label="Add Embedding"
      tooltip="Add Embedding"
      icon={<BiCode />}
      sx={{
        p: 2,
        color: 'base.700',
        _hover: {
          color: 'base.550',
        },
        _active: {
          color: 'base.500',
        },
      }}
      variant="link"
      onClick={onClick}
    />
  );
};

export default memo(AddEmbeddingButton);
