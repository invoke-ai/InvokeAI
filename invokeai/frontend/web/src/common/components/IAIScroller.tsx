import { Box, BoxProps, Flex } from '@chakra-ui/react';
import OverlayScrollable from 'features/ui/components/common/OverlayScrollable';
import { PropsWithChildren, memo } from 'react';

type IAIScrollerProps = PropsWithChildren & BoxProps;

const IAIScroller = (props: IAIScrollerProps) => {
  const { ...rest } = props;
  return (
    <Box
      sx={{
        position: 'relative',
        flexShrink: 0,
      }}
      {...rest}
    >
      <OverlayScrollable>
        <Flex position="absolute" width="100%">
          {props.children}
        </Flex>
      </OverlayScrollable>
    </Box>
  );
};

export default memo(IAIScroller);
