import { Box, Divider, Heading, Text } from '@invoke-ai/ui-library';
import ScrollableContent from 'common/components/OverlayScrollbars/ScrollableContent';
import { InputFieldGate } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldGate';
import { InputFieldViewMode } from 'features/nodes/components/flow/nodes/Invocation/fields/InputFieldViewMode';
import {
  type BuilderElement,
  type ContainerElement,
  data,
  type DividerElement,
  type FieldElement,
  type HeadingElement,
  type NotesElement,
} from 'features/nodes/types/workflow';
import { createContext, memo, useContext, useMemo } from 'react';
import { assert } from 'tsafe';

const ContainerContext = createContext<{ orientation: ContainerElement['orientation']; depth: number } | null>(null);

const useContainerContextContext = () => {
  const context = useContext(ContainerContext);
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
    case 'container':
      return <ContainerElementComponent element={element} />;
    case 'field':
      return <FieldElementComponent element={element} />;
    case 'heading':
      return <HeadingElementComponent element={element} />;
    case 'notes':
      return <NotesElementComponent element={element} />;
    case 'divider':
      return <DividerElementComponent element={element} />;
    default:
      assert(false, `Unhandled element type: ${element}`);
  }
};

const ContainerElementComponent = ({ element }: { element: ContainerElement }) => {
  const { children, orientation } = element;

  const parentCtx = useContext(ContainerContext);
  const depth = useMemo(() => (parentCtx ? parentCtx.depth + 1 : 0), [parentCtx]);
  const ctx = useMemo(() => ({ orientation, depth }), [depth, orientation]);

  const gridAutoX = useMemo(() => {
    return children
      .map(({ type }) => {
        switch (type) {
          case 'divider':
            return 'min-content';
          case 'notes':
          case 'heading':
          case 'container':
          case 'field':
            return 'auto';
        }
      })
      .join(' ');
  }, [children]);

  if (orientation === 'horizontal') {
    return (
      <ContainerContext.Provider value={ctx}>
        <Box id={element.id} display="grid" gridAutoFlow="column" gridAutoColumns={gridAutoX} gap={2} overflow="hidden">
          {children.map((child) => (
            <ElementComponent key={child.id} element={child} />
          ))}
        </Box>
      </ContainerContext.Provider>
    );
  }

  // orientation === 'vertical'
  return (
    <ContainerContext.Provider value={ctx}>
      <Box id={element.id} display="grid" gridAutoFlow="row" gridAutoRows={gridAutoX} gap={2} overflow="hidden">
        {children.map((child) => (
          <ElementComponent key={child.id} element={child} />
        ))}
      </Box>
    </ContainerContext.Provider>
  );
};

const FieldElementComponent = ({ element }: { element: FieldElement }) => {
  const { fieldIdentifier } = element;

  return (
    <InputFieldGate nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName}>
      <InputFieldViewMode nodeId={fieldIdentifier.nodeId} fieldName={fieldIdentifier.fieldName} />
    </InputFieldGate>
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
  const { content, level } = element;

  return (
    <Heading id={element.id} size={LEVEL_TO_SIZE[level]}>
      {content}
    </Heading>
  );
};

const NotesElementComponent = ({ element }: { element: NotesElement }) => {
  const { content, fontSize } = element;

  return (
    <Text id={element.id} fontSize={fontSize}>
      {content}
    </Text>
  );
};

const DividerElementComponent = ({ element }: { element: DividerElement }) => {
  const { orientation } = useContainerContextContext();

  if (orientation === 'horizontal') {
    return <Divider id={element.id} orientation="vertical" />;
  }

  // orientation === 'vertical'
  return <Divider id={element.id} orientation="horizontal" />;
};
