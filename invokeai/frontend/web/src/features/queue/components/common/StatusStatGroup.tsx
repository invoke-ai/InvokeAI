import { StatGroup, StatGroupProps } from '@chakra-ui/react';
import { memo } from 'react';

const StatusStatGroup = ({ children, ...rest }: StatGroupProps) => (
  <StatGroup
    alignItems="center"
    justifyContent="center"
    w="full"
    h="full"
    layerStyle="second"
    borderRadius="base"
    py={2}
    px={3}
    gap={6}
    flexWrap="nowrap"
    {...rest}
  >
    {children}
  </StatGroup>
);

export default memo(StatusStatGroup);
