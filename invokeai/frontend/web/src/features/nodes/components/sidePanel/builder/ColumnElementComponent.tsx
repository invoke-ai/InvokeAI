import { Flex, Grid, GridItem } from '@invoke-ai/ui-library';
import {
  ContainerElementComponent,
  useContainerContext,
} from 'features/nodes/components/sidePanel/builder/ContainerElementComponent';
import { DividerElementComponent } from 'features/nodes/components/sidePanel/builder/DividerElementComponent';
import { HeadingElementComponent } from 'features/nodes/components/sidePanel/builder/HeadingElementComponent';
import { NodeFieldElementComponent } from 'features/nodes/components/sidePanel/builder/NodeFieldElementComponent';
import { TextElementComponent } from 'features/nodes/components/sidePanel/builder/TextElementComponent';
import type { ColumnChildElement, ColumnElement } from 'features/nodes/types/workflow';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext, useMemo } from 'react';
import type { Equals } from 'tsafe';
import { assert } from 'tsafe';

const _ColumnContext = createContext<{ columnId: string; columnNumber: number } | null>(null);
const ColumnContextProvider = ({
  columnId,
  columnNumber,
  children,
}: PropsWithChildren<{ columnId: string; columnNumber: number }>) => {
  const ctx = useMemo(() => ({ columnId, columnNumber }), [columnId, columnNumber]);
  return <_ColumnContext.Provider value={ctx}>{children}</_ColumnContext.Provider>;
};
export const useColumnContext = () => {
  const context = useContext(_ColumnContext);
  assert(context !== null);
  return context;
};

const ColumnElementChildComponent = memo(({ element }: { element: ColumnChildElement }) => {
  const { type, id } = element;
  switch (type) {
    case 'container':
      return <ContainerElementComponent key={id} element={element} />;
    case 'node-field':
      return <NodeFieldElementComponent key={id} element={element} />;
    case 'divider':
      return <DividerElementComponent key={id} element={element} />;
    case 'heading':
      return <HeadingElementComponent key={id} element={element} />;
    case 'text':
      return <TextElementComponent key={id} element={element} />;
    default:
      assert<Equals<typeof type, never>>(false, `Unhandled type ${type}`);
  }
});
ColumnElementChildComponent.displayName = 'ColumnElementChildComponent';

export const ColumnElementComponent = memo(({ element }: { element: ColumnElement }) => {
  const containerCtx = useContainerContext();
  const columnNumber = useMemo(
    () => containerCtx.columnIds.indexOf(element.id) + 1,
    [containerCtx.columnIds, element.id]
  );
  const withDivider = useMemo(
    () => containerCtx.columnIds.indexOf(element.id) + 1 < containerCtx.columnIds.length,
    [containerCtx.columnIds, element.id]
  );
  return (
    <ColumnContextProvider columnId={element.id} columnNumber={columnNumber}>
      <>
        <GridItem
          as={Grid}
          id={`column:${element.id}_${columnNumber}`}
          gap={4}
          gridAutoRows="min-content"
          gridAutoFlow="row"
        >
          {element.data.elements.map((element) => (
            <ColumnElementChildComponent key={element.id} element={element} />
          ))}
        </GridItem>
        {withDivider && <Flex w="1px" bg="base.800" flexShrink={0} />}
      </>
    </ColumnContextProvider>
  );
});

ColumnElementComponent.displayName = 'ColumnElementComponent';
