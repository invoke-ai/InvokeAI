import { Box, ChakraProps } from '@chakra-ui/react';
import { useOverlayScrollbars } from 'overlayscrollbars-react';
import { ReactNode, useEffect, useRef } from 'react';

type ScrollableProps = {
  children: ReactNode;
  containerProps?: ChakraProps;
};

const Scrollable = ({
  children,
  containerProps = {
    width: 'full',
    height: 'full',
    flexShrink: 0,
  },
}: ScrollableProps) => {
  const scrollableRef = useRef<HTMLDivElement>(null);
  const topShadowRef = useRef<HTMLDivElement>(null);
  const bottomShadowRef = useRef<HTMLDivElement>(null);

  const [initialize, _instance] = useOverlayScrollbars({
    defer: true,
    events: {
      initialized(instance) {
        if (!topShadowRef.current || !bottomShadowRef.current) {
          return;
        }

        const { scrollTop, scrollHeight, offsetHeight } =
          instance.elements().content;

        const scrollPercentage = scrollTop / (scrollHeight - offsetHeight);

        topShadowRef.current.style.opacity = String(scrollPercentage * 5);

        bottomShadowRef.current.style.opacity = String(
          (1 - scrollPercentage) * 5
        );
      },
      scroll: (_instance, event) => {
        if (
          !topShadowRef.current ||
          !bottomShadowRef.current ||
          !scrollableRef.current
        ) {
          return;
        }

        const { scrollTop, scrollHeight, offsetHeight } =
          event.target as HTMLDivElement;

        const scrollPercentage = scrollTop / (scrollHeight - offsetHeight);

        topShadowRef.current.style.opacity = String(scrollPercentage * 5);

        bottomShadowRef.current.style.opacity = String(
          (1 - scrollPercentage) * 5
        );
      },
    },
  });

  useEffect(() => {
    if (
      !scrollableRef.current ||
      !topShadowRef.current ||
      !bottomShadowRef.current
    ) {
      return;
    }

    topShadowRef.current.style.opacity = '0';

    bottomShadowRef.current.style.opacity = '0';

    initialize(scrollableRef.current);
  }, [initialize]);

  return (
    <Box position="relative" w="full" h="full">
      <Box ref={scrollableRef} {...containerProps} overflowY="scroll">
        <Box paddingInlineEnd={5}>{children}</Box>
      </Box>
      <Box
        ref={bottomShadowRef}
        sx={{
          position: 'absolute',
          boxShadow:
            'inset 0 -3.5rem 2rem -2rem var(--invokeai-colors-base-900)',
          width: 'full',
          height: 24,
          bottom: 0,
          left: 0,
          pointerEvents: 'none',
          transition: 'opacity 0.2s',
        }}
      ></Box>
      <Box
        ref={topShadowRef}
        sx={{
          position: 'absolute',
          boxShadow:
            'inset 0 3.5rem 2rem -2rem var(--invokeai-colors-base-900)',
          width: 'full',
          height: 24,
          top: 0,
          left: 0,
          pointerEvents: 'none',
          transition: 'opacity 0.2s',
        }}
      ></Box>
    </Box>
  );
};

export default Scrollable;
