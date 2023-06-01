import IAIButton from 'common/components/IAIButton';
import { memo } from 'react';

type ControlNetProcessButtonProps = {
  onClick: () => void;
};

const ControlNetProcessButton = (props: ControlNetProcessButtonProps) => {
  const { onClick } = props;
  return <IAIButton onClick={onClick}>Process Control Image</IAIButton>;
};

export default memo(ControlNetProcessButton);
