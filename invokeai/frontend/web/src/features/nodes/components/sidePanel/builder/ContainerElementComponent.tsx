import { Grid } from '@invoke-ai/ui-library';
import { ColumnElementComponent } from 'features/nodes/components/sidePanel/builder/ColumnElementComponent';
import type { ContainerElement } from 'features/nodes/types/workflow';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext, useMemo } from 'react';
import { assert } from 'tsafe';

const _ContainerContext = createContext<{ containerId: string; columnIds: string[]; depth: number } | null>(null);
const ContainerContextProvider = ({
  containerId,
  columnIds,
  children,
}: PropsWithChildren<{ containerId: string; columnIds: string[] }>) => {
  const parentCtx = useContext(_ContainerContext);
  const ctx = useMemo(
    () => ({ containerId, columnIds, depth: parentCtx ? parentCtx.depth + 1 : 0 }),
    [columnIds, containerId, parentCtx]
  );
  return <_ContainerContext.Provider value={ctx}>{children}</_ContainerContext.Provider>;
};
export const useContainerContext = () => {
  const context = useContext(_ContainerContext);
  assert(context !== null);
  return context;
};

const getGridTemplateColumns = (count: number) => {
  return Array.from({ length: count }, () => '1fr').join(' auto ');
};

export const ContainerElementComponent = memo(({ element }: { element: ContainerElement }) => {
  const { id, data } = element;
  const { columns } = data;
  const columnIds = useMemo(() => columns.map((column) => column.id), [columns]);

  return (
    <ContainerContextProvider containerId={id} columnIds={columnIds}>
      <Grid id={id} gap={4} gridTemplateColumns={getGridTemplateColumns(columns.length)} gridAutoFlow="column">
        {columns.map((element, i) => {
          return <ColumnElementComponent key={`column:${id}_${i + 1}`} element={element} />;
        })}
      </Grid>
    </ContainerContextProvider>
  );
});
ContainerElementComponent.displayName = 'ContainerElementComponent';
