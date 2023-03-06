import { Flex } from '@chakra-ui/react';
import { ReactNode } from 'react';

type WorkInProgressProps = {
  children: ReactNode;
};

const WorkInProgress = (props: WorkInProgressProps) => {
  const { children } = props;

  return (
    <Flex
      sx={{
        width: '100%',
        height: '100%',
        bg: 'base.850',
      }}
    >
      {children}
    </Flex>
  );
};

export default WorkInProgress;
