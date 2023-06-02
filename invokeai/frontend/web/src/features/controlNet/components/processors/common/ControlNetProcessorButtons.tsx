import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import { FaUndo } from 'react-icons/fa';
import IAIButton from 'common/components/IAIButton';

type ControlNetProcessorButtonsProps = {
  handleProcess: () => void;
  isProcessDisabled: boolean;
  handleReset: () => void;
  isResetDisabled: boolean;
};

const ControlNetProcessorButtons = (props: ControlNetProcessorButtonsProps) => {
  const { handleProcess, isProcessDisabled, handleReset, isResetDisabled } =
    props;
  return (
    <Flex
      sx={{
        gap: 4,
        w: 'full',
        alignItems: 'center',
        justifyContent: 'stretch',
      }}
    ></Flex>
  );
};

export default memo(ControlNetProcessorButtons);
