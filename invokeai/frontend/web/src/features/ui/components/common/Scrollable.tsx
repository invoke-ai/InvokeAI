import { Box, ChakraProps } from '@chakra-ui/react';
import { throttle } from 'lodash';
import { ReactNode, useEffect, useRef } from 'react';

const scrollShadowBaseStyles: ChakraProps['sx'] = {
  position: 'absolute',
  width: 'full',
  height: 24,
  left: 0,
  pointerEvents: 'none',
  transition: 'opacity 0.2s ease-in-out',
};

type ScrollableProps = {
  children: ReactNode;
};

const Scrollable = ({ children }: ScrollableProps) => {
  const scrollableRef = useRef<HTMLDivElement>(null);
  const topShadowRef = useRef<HTMLDivElement>(null);
  const bottomShadowRef = useRef<HTMLDivElement>(null);

  const updateScrollShadowOpacity = throttle(
    () => {
      if (
        !scrollableRef.current ||
        !topShadowRef.current ||
        !bottomShadowRef.current
      ) {
        return;
      }
      const { scrollTop, scrollHeight, offsetHeight } = scrollableRef.current;

      if (scrollTop > 0) {
        topShadowRef.current.style.opacity = '1';
      } else {
        topShadowRef.current.style.opacity = '0';
      }

      if (scrollTop >= scrollHeight - offsetHeight) {
        bottomShadowRef.current.style.opacity = '0';
      } else {
        bottomShadowRef.current.style.opacity = '1';
      }
    },
    33,
    { leading: true }
  );

  useEffect(() => {
    updateScrollShadowOpacity();
  }, [updateScrollShadowOpacity]);

  return (
    <Box position="relative" w="full" h="full">
      <Box
        ref={scrollableRef}
        position="absolute"
        w="full"
        h="full"
        overflowY="scroll"
        onScroll={updateScrollShadowOpacity}
      >
        {children}
      </Box>
      <Box
        ref={bottomShadowRef}
        sx={{
          ...scrollShadowBaseStyles,
          bottom: 0,
          boxShadow:
            'inset 0 -3.5rem 2rem -2rem var(--invokeai-colors-base-900)',
        }}
      ></Box>
      <Box
        ref={topShadowRef}
        sx={{
          ...scrollShadowBaseStyles,
          top: 0,
          boxShadow:
            'inset 0 3.5rem 2rem -2rem var(--invokeai-colors-base-900)',
        }}
      ></Box>
    </Box>
  );
};

export default Scrollable;
