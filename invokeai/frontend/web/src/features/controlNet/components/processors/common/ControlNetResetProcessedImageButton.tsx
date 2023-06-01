import IAIButton from 'common/components/IAIButton';
import { memo } from 'react';
import { FaUnderline, FaUndo } from 'react-icons/fa';

type ControlNetResetProcessedImageButtonProps = {
  onClick: () => void;
};

const ControlNetResetProcessedImageButton = (
  props: ControlNetResetProcessedImageButtonProps
) => {
  const { onClick } = props;
  return (
    <IAIButton leftIcon={<FaUndo />} onClick={onClick}>
      Reset Processing
    </IAIButton>
  );
};

export default memo(ControlNetResetProcessedImageButton);
