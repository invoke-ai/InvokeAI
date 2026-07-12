import type { BoxProps, FlexProps } from '@chakra-ui/react';

import { Box, Flex } from '@chakra-ui/react';
import { Panel } from '@workbench/components/ui';

type Operation = 'filter' | 'select-object';
const RESPONSIVE_PANEL_WIDTH = { base: '26rem', md: '30rem' } as const;

const Root = ({ children, operation, ...rest }: BoxProps & { operation: Operation }) => (
  <Panel
    data-operation={operation}
    data-operation-panel-max-width="container"
    data-operation-panel-width="responsive"
    density="none"
    display="flex"
    flexDirection="column"
    maxH="min(42rem, calc(100dvh - 2rem))"
    maxW="100%"
    minH="0"
    overflow="hidden"
    pointerEvents="auto"
    role="region"
    rounded="xl"
    shadow="lg"
    tone="raised"
    w={RESPONSIVE_PANEL_WIDTH}
    {...rest}
  >
    {children}
  </Panel>
);

const Header = (props: BoxProps) => (
  <Box as="header" borderBottomWidth="1px" data-slot="header" flexShrink="0" px="4" py="3" {...props} />
);

const Body = (props: BoxProps) => (
  <Box
    data-scroll-container="body"
    data-slot="body"
    flex="1"
    minH="0"
    overflowX="hidden"
    overflowY="auto"
    px="4"
    py="3"
    {...props}
  />
);

const Feedback = (props: BoxProps) => (
  <Box borderTopWidth="1px" data-slot="feedback" flexShrink="0" minH="16" px="4" py="2" {...props} />
);

const Footer = (props: FlexProps) => (
  <Flex
    as="footer"
    borderTopWidth="1px"
    data-slot="footer"
    flexShrink="0"
    flexWrap="wrap"
    gap="2"
    px="4"
    py="3"
    {...props}
  />
);

export const CanvasOperationPanel = { Body, Feedback, Footer, Header, Root };
