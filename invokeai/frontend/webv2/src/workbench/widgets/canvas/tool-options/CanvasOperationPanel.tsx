import type { BoxProps, FlexProps } from '@chakra-ui/react';

import { Box, Flex } from '@chakra-ui/react';
import { Panel } from '@workbench/components/ui';

type Operation = 'filter' | 'select-object';

export const CANVAS_OPERATION_PANEL_LAYOUT = {
  flex: '0 1 30rem',
  maxH: 'full',
  maxW: 'full',
  minH: '0',
  minW: '0',
  overflow: 'hidden',
  w: '30rem',
} satisfies BoxProps;

export const CANVAS_OPERATION_FOOTER_LAYOUT = {
  flexWrap: 'wrap',
  minW: '0',
} satisfies FlexProps;

export const CANVAS_OPERATION_BODY_LAYOUT = {
  flex: '1',
  minH: '0',
  overflowX: 'hidden',
  overflowY: 'auto',
} satisfies BoxProps;

export const CANVAS_OPERATION_FIXED_SECTION_LAYOUT = { flexShrink: '0' } satisfies BoxProps;

const Root = ({ children, operation, ...rest }: BoxProps & { operation: Operation }) => (
  <Panel
    data-operation={operation}
    density="none"
    display="flex"
    flexDirection="column"
    pointerEvents="auto"
    role="region"
    rounded="xl"
    shadow="lg"
    tone="raised"
    {...CANVAS_OPERATION_PANEL_LAYOUT}
    {...rest}
  >
    {children}
  </Panel>
);

const Header = (props: BoxProps) => (
  <Box
    as="header"
    borderBottomWidth="1px"
    data-slot="header"
    px="4"
    py="3"
    {...CANVAS_OPERATION_FIXED_SECTION_LAYOUT}
    {...props}
  />
);

const Body = (props: BoxProps) => (
  <Box data-scroll-container="body" data-slot="body" px="4" py="3" {...CANVAS_OPERATION_BODY_LAYOUT} {...props} />
);

const Feedback = (props: BoxProps) => (
  <Box
    borderTopWidth="1px"
    data-slot="feedback"
    minH="16"
    px="4"
    py="2"
    {...CANVAS_OPERATION_FIXED_SECTION_LAYOUT}
    {...props}
  />
);

const Footer = (props: FlexProps) => (
  <Flex
    as="footer"
    borderTopWidth="1px"
    data-slot="footer"
    gap="2"
    px="4"
    py="3"
    {...CANVAS_OPERATION_FOOTER_LAYOUT}
    {...CANVAS_OPERATION_FIXED_SECTION_LAYOUT}
    {...props}
  />
);

export const CanvasOperationPanel = { Body, Feedback, Footer, Header, Root };
