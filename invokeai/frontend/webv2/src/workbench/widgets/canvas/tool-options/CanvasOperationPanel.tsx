import type { BoxProps, FlexProps } from '@chakra-ui/react';

import { Box, Flex } from '@chakra-ui/react';
import { Panel } from '@workbench/components/ui';
import { createContext, useContext } from 'react';

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

export const CANVAS_SELECT_OBJECT_PANEL_LAYOUT = {
  flex: '0 1 26.25rem',
  maxH: 'full',
  maxW: 'full',
  minH: '0',
  minW: '0',
  overflow: 'hidden',
  w: '26.25rem',
} satisfies BoxProps;

export const CANVAS_SELECT_OBJECT_SLOT_LAYOUT = {
  px: '3',
  py: '2',
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

const OperationContext = createContext<Operation>('filter');

const Root = ({ children, operation, ...rest }: BoxProps & { operation: Operation }) => {
  const layout = operation === 'select-object' ? CANVAS_SELECT_OBJECT_PANEL_LAYOUT : CANVAS_OPERATION_PANEL_LAYOUT;
  return (
    <OperationContext value={operation}>
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
        {...layout}
        {...rest}
      >
        {children}
      </Panel>
    </OperationContext>
  );
};

const Header = (props: BoxProps) => {
  const operation = useContext(OperationContext);
  return (
    <Box
      as="header"
      borderBottomWidth={operation === 'select-object' ? undefined : '1px'}
      data-slot="header"
      {...CANVAS_SELECT_OBJECT_SLOT_LAYOUT}
      {...CANVAS_OPERATION_FIXED_SECTION_LAYOUT}
      {...props}
    />
  );
};

const Body = (props: BoxProps) => {
  const operation = useContext(OperationContext);
  return (
    <Box
      data-scroll-container="body"
      data-slot="body"
      {...CANVAS_SELECT_OBJECT_SLOT_LAYOUT}
      {...CANVAS_OPERATION_BODY_LAYOUT}
      flex={operation === 'select-object' ? '0 1 auto' : CANVAS_OPERATION_BODY_LAYOUT.flex}
      {...props}
    />
  );
};

const Feedback = (props: BoxProps) => {
  const operation = useContext(OperationContext);
  return (
    <Box
      borderTopWidth={operation === 'select-object' ? undefined : '1px'}
      data-slot="feedback"
      minH={operation === 'select-object' ? undefined : '16'}
      px="4"
      py="2"
      {...CANVAS_OPERATION_FIXED_SECTION_LAYOUT}
      {...props}
    />
  );
};

const Footer = (props: FlexProps) => (
  <Flex
    as="footer"
    borderTopWidth="1px"
    data-slot="footer"
    gap="2"
    {...CANVAS_SELECT_OBJECT_SLOT_LAYOUT}
    {...CANVAS_OPERATION_FOOTER_LAYOUT}
    {...CANVAS_OPERATION_FIXED_SECTION_LAYOUT}
    {...props}
  />
);

export const CanvasOperationPanel = { Body, Feedback, Footer, Header, Root };
