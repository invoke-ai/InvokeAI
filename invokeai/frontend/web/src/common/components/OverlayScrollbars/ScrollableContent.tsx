import { combine } from '@atlaskit/pragmatic-drag-and-drop/combine';
import { autoScrollForElements } from '@atlaskit/pragmatic-drag-and-drop-auto-scroll/element';
import { autoScrollForExternal } from '@atlaskit/pragmatic-drag-and-drop-auto-scroll/external';
import type { ChakraProps } from '@invoke-ai/ui-library';
import { Box, Flex } from '@invoke-ai/ui-library';
import { getOverlayScrollbarsParams } from 'common/components/OverlayScrollbars/constants';
import type { OverlayScrollbarsComponentRef } from 'overlayscrollbars-react';
import { OverlayScrollbarsComponent } from 'overlayscrollbars-react';
import type { CSSProperties, PropsWithChildren } from 'react';
import { memo, useEffect, useMemo, useState } from 'react';

type Props = PropsWithChildren & {
  maxHeight?: ChakraProps['maxHeight'];
  overflowX?: 'hidden' | 'scroll';
  overflowY?: 'hidden' | 'scroll';
};

const styles: CSSProperties = { position: 'absolute', top: 0, left: 0, right: 0, bottom: 0 };

const ScrollableContent = ({ children, maxHeight, overflowX = 'hidden', overflowY = 'scroll' }: Props) => {
  const overlayscrollbarsOptions = useMemo(
    () => getOverlayScrollbarsParams(overflowX, overflowY).options,
    [overflowX, overflowY]
  );
  const [os, osRef] = useState<OverlayScrollbarsComponentRef | null>(null);
  useEffect(() => {
    const osInstance = os?.osInstance();

    if (!osInstance) {
      return;
    }

    const element = osInstance.elements().viewport;

    // `pragmatic-drag-and-drop-auto-scroll` requires the element to have `overflow-y: scroll` or `overflow-y: auto`
    // else it logs an ugly warning. In our case, using a custom scrollbar library, it will be 'hidden' by default.
    // To prevent the erroneous warning, we temporarily set the overflow-y to 'scroll' and then revert it back.
    const overflowY = element.style.overflowY; // starts 'hidden'
    element.style.setProperty('overflow-y', 'scroll', 'important');
    const cleanup = combine(autoScrollForElements({ element }), autoScrollForExternal({ element }));
    element.style.setProperty('overflow-y', overflowY);

    return cleanup;
  }, [os]);

  return (
    <Flex w="full" h="full" maxHeight={maxHeight} position="relative">
      <Box position="absolute" top={0} left={0} right={0} bottom={0}>
        <OverlayScrollbarsComponent ref={osRef} style={styles} options={overlayscrollbarsOptions}>
          {children}
        </OverlayScrollbarsComponent>
      </Box>
    </Flex>
  );
};

export default memo(ScrollableContent);
