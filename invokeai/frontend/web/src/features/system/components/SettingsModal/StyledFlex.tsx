import { Flex } from '@chakra-ui/react';
import { PropsWithChildren, memo } from 'react';

const StyledFlex = (props: PropsWithChildren) => {
  return (
    <Flex
      sx={{
        flexDirection: 'column',
        gap: 2,
        p: 4,
        borderRadius: 'base',
        bg: 'base.100',
        _dark: {
          bg: 'base.900',
        },
      }}
    >
      {props.children}
    </Flex>
  );
};

export default memo(StyledFlex);
