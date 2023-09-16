import { ButtonGroup, ButtonGroupProps, Flex } from '@chakra-ui/react';
import { memo } from 'react';
import CancelQueueButton from './CancelQueueButton';
import ClearQueueButton from './ClearQueueButton';
import PruneQueueButton from './PruneQueueButton';
import StartQueueButton from './StartQueueButton';
import StopQueueButton from './StopQueueButton';

type Props = ButtonGroupProps & {
  asIconButtons?: boolean;
};

const VerticalQueueControls = ({ asIconButtons, ...rest }: Props) => {
  return (
    <Flex flexDir="column" gap={2}>
      <ButtonGroup w="full" isAttached {...rest}>
        <StartQueueButton asIconButton={asIconButtons} />
        <StopQueueButton asIconButton={asIconButtons} />
        <CancelQueueButton asIconButton={asIconButtons} />
      </ButtonGroup>
      <ButtonGroup w="full" isAttached {...rest}>
        <PruneQueueButton asIconButton={asIconButtons} />
        <ClearQueueButton asIconButton={asIconButtons} />
      </ButtonGroup>
    </Flex>
  );
};

export default memo(VerticalQueueControls);
