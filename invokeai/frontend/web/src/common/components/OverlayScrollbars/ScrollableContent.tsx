import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, Flex } from '@invoke-ai/ui-library';
import { getOverlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties, PropsWithChildren } from 'react';
import { memo, useMemo } from 'react';

type Props = PropsWithChildren & {
  maxHeight?: ChakraProps['maxHeight'];
  overflowX?: 'hidden' | 'scroll';
  overflowY?: 'hidden' | 'scroll';
};

const styles: CSSProperties = { height: '100%', width: '100%' };

const ScrollableContent = ({ children, maxHeight, overflowX = 'hidden', overflowY = 'scroll' }: Props) => {
  const overlayscrollbarsOptions = useMemo(
    () => getOverlayScrollbarsParams(overflowX, overflowY).options,
    [overflowX, overflowY]
  );
  return (
    <Flex w="full" h="full" maxHeight={maxHeight} position="relative">
      <Box position="absolute" top={0} left={0} right={0} bottom={0}>
        <OverlayScrollbarsComponent defer style={styles} options={overlayscrollbarsOptions}>
          {children}
        </OverlayScrollbarsComponent>
      </Box>
    </Flex>
  );
};

export default memo(ScrollableContent);
