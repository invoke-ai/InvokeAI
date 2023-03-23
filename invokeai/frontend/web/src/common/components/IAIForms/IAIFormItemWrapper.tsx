import { Flex } from '@chakra-ui/react';
import { ReactElement } from 'react';

export function IAIFormItemWrapper({
  children,
}: {
  children: ReactElement | ReactElement[];
}) {
  return (
    <Flex
      sx={{
        flexDirection: 'column',
        padding: 4,
        rowGap: 4,
        borderRadius: 'base',
        width: 'full',
        bg: 'base.900',
      }}
    >
      {children}
    </Flex>
  );
}
