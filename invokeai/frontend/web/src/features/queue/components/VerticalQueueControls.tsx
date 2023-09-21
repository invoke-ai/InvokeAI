import { ButtonGroup, ButtonGroupProps, Flex } from '@chakra-ui/react';
import { memo } from 'react';
import ClearQueueButton from './ClearQueueButton';
import PauseProcessorButton from './PauseProcessorButton';
import PruneQueueButton from './PruneQueueButton';
import ResumeProcessorButton from './ResumeProcessorButton';

type Props = ButtonGroupProps & {
  asIconButtons?: boolean;
};

const VerticalQueueControls = ({ asIconButtons, ...rest }: Props) => {
  return (
    <Flex flexDir="column" gap={2}>
      <ButtonGroup w="full" isAttached {...rest}>
        <ResumeProcessorButton asIconButton={asIconButtons} />
        <PauseProcessorButton asIconButton={asIconButtons} />
      </ButtonGroup>
      <ButtonGroup w="full" isAttached {...rest}>
        <PruneQueueButton asIconButton={asIconButtons} />
        <ClearQueueButton asIconButton={asIconButtons} />
      </ButtonGroup>
    </Flex>
  );
};

export default memo(VerticalQueueControls);
