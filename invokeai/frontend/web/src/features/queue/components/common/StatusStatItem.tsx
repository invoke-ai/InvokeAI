import type { ChakraProps, StatProps } from '@chakra-ui/react';
import { Stat, StatLabel, StatNumber } from '@chakra-ui/react';
import { memo } from 'react';

const sx: ChakraProps['sx'] = {
  '&[aria-disabled="true"]': {
    color: 'base.500',
  },
};

type Props = Omit<StatProps, 'children'> & {
  label: string;
  value: string | number;
  isDisabled?: boolean;
};

const StatusStatItem = ({
  label,
  value,
  isDisabled = false,
  ...rest
}: Props) => (
  <Stat
    flexGrow={1}
    textOverflow="ellipsis"
    overflow="hidden"
    whiteSpace="nowrap"
    aria-disabled={isDisabled}
    sx={sx}
    {...rest}
  >
    <StatLabel textOverflow="ellipsis" overflow="hidden" whiteSpace="nowrap">
      {label}
    </StatLabel>
    <StatNumber>{value}</StatNumber>
  </Stat>
);

export default memo(StatusStatItem);
