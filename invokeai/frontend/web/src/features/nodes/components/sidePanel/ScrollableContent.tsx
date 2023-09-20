import { Box, Flex, StyleProps } from '@chakra-ui/react';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import { PropsWithChildren, memo } from 'react';

type Props = PropsWithChildren & {
  maxHeight?: StyleProps['maxHeight'];
};

const ScrollableContent = ({ children, maxHeight }: Props) => {
  return (
    <Flex
      sx={{
        w: 'full',
        h: 'full',
        maxHeight,
        position: 'relative',
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
        }}
      >
        <OverlayScrollbarsComponent
          defer
          style={{ height: '100%', width: '100%' }}
          options={{
            scrollbars: {
              visibility: 'auto',
              autoHide: 'scroll',
              autoHideDelay: 1300,
              theme: 'os-theme-dark',
            },
            overflow: {
              x: 'hidden',
            },
          }}
        >
          {children}
        </OverlayScrollbarsComponent>
      </Box>
    </Flex>
  );
};

export default memo(ScrollableContent);
