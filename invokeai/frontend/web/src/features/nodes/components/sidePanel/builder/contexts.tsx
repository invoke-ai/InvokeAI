import type { ContainerElement, ElementId } from 'features/nodes/types/workflow';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext, useMemo } from 'react';

type ContainerContextValue = {
  id: ElementId;
  direction: ContainerElement['data']['direction'];
};

const ContainerContext = createContext<ContainerContextValue | null>(null);

export const ContainerContextProvider = memo(
  ({ id, direction, children }: PropsWithChildren<ContainerContextValue>) => {
    const ctxValue = useMemo(() => ({ id, direction }), [id, direction]);
    return <ContainerContext.Provider value={ctxValue}>{children}</ContainerContext.Provider>;
  }
);
ContainerContextProvider.displayName = 'ContainerContextProvider';

export const useContainerContext = () => {
  const container = useContext(ContainerContext);
  return container;
};

const DepthContext = createContext<number>(0);

export const DepthContextProvider = memo(({ depth, children }: PropsWithChildren<{ depth: number }>) => {
  return <DepthContext.Provider value={depth}>{children}</DepthContext.Provider>;
});
DepthContextProvider.displayName = 'DepthContextProvider';

export const useDepthContext = () => {
  const depth = useContext(DepthContext);
  return depth;
};
