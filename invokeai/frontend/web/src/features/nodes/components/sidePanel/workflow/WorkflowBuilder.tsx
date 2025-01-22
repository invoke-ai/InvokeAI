import { Flex, Heading, Text } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { InputFieldViewMode } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldViewMode';
import type {
  BuilderElement,
  DividerElement,
  FieldElement,
  HeadingElement,
  StackElement,
  TextElement,
} from 'features/nodes/types/workflow';
import { data } from 'features/nodes/types/workflow';
import { createContext, memo, useContext, useMemo } from 'react';
import { assert } from 'tsafe';

const StackContext = createContext<{ direction: StackElement['data']['direction']; depth: number } | null>(null);

const useStackContext = () => {
  const context = useContext(StackContext);
  assert(context !== null);
  return context;
};

export const WorkflowBuilder = memo(() => {
  return (
    <ScrollableContent>
      <ElementComponent element={data} />
    </ScrollableContent>
  );
});

WorkflowBuilder.displayName = 'WorkflowBuilder';

const ElementComponent = ({ element }: { element: BuilderElement }) => {
  switch (element.type) {
    case 'stack':
      return <StackElementComponent element={element} />;
    case 'field':
      return <FieldElementComponent element={element} />;
    case 'heading':
      return <HeadingElementComponent element={element} />;
    case 'text':
      return <TextElementComponent element={element} />;
    case 'divider':
      return <DividerElementComponent element={element} />;
    default:
      assert(false, `Unhandled element type: ${element}`);
  }
};

const DIRECTION_TO_FLEXDIR = {
  horizontal: 'row',
  vertical: 'column',
} as const;
const StackElementComponent = ({ element }: { element: StackElement }) => {
  const { id, data } = element;
  const { children, direction } = data;

  const parentCtx = useContext(StackContext);
  const depth = useMemo(() => (parentCtx ? parentCtx.depth + 1 : 0), [parentCtx]);
  const ctx = useMemo(() => ({ direction, depth }), [depth, direction]);

  return (
    <StackContext.Provider value={ctx}>
      <Flex id={id} gap={2} flexDir={DIRECTION_TO_FLEXDIR[direction]}>
        {children.map((child) => (
          <ElementComponent key={child.id} element={child} />
        ))}
      </Flex>
    </StackContext.Provider>
  );
};

const FieldElementComponent = ({ element }: { element: FieldElement }) => {
  const { id, data } = element;
  const { fieldIdentifier } = data;

  return (
    <Flex id={id} flexBasis="100%">
      <InputFieldGate nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName}>
        <InputFieldViewMode nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName} />
      </InputFieldGate>
    </Flex>
  );
};

const LEVEL_TO_SIZE = {
  1: 'xl',
  2: 'lg',
  3: 'md',
  4: 'sm',
  5: 'xs',
} as const;
const HeadingElementComponent = ({ element }: { element: HeadingElement }) => {
  const { id, data } = element;
  const { content, level } = data;

  return (
    <Heading id={id} size={LEVEL_TO_SIZE[level]}>
      {content}
    </Heading>
  );
};

const TextElementComponent = ({ element }: { element: TextElement }) => {
  const { id, data } = element;
  const { content, fontSize } = data;

  return (
    <Text id={id} fontSize={fontSize}>
      {content}
    </Text>
  );
};

const DIRECTION_TO_WIDTH = {
  horizontal: '1px',
  vertical: undefined,
};

const DIRECTION_TO_HEIGHT = {
  horizontal: undefined,
  vertical: '1px',
};

const DividerElementComponent = ({ element }: { element: DividerElement }) => {
  const { id } = element;
  const { direction } = useStackContext();

  return (
    <Flex id={id} w={DIRECTION_TO_WIDTH[direction]} h={DIRECTION_TO_HEIGHT[direction]} bg="base.700" flexShrink={0} />
  );
};
