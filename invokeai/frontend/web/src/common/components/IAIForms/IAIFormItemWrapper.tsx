import { Flex, useColorMode } from '@chakra-ui/react';
import { ReactElement } from 'react';
import { mode } from 'theme/util/mode';

export function IAIFormItemWrapper({
  children,
}: {
  children: ReactElement | ReactElement[];
}) {
  const { colorMode } = useColorMode();
  return (
    <Flex
      sx={{
        flexDirection: 'column',
        padding: 4,
        rowGap: 4,
        borderRadius: 'base',
        width: 'full',
        bg: mode('base.100', 'base.900')(colorMode),
      }}
    >
      {children}
    </Flex>
  );
}
