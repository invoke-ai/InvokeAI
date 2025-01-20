import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext } from 'react';
import { assert } from 'tsafe';

type ViewType = 'linear-user' | 'linear-editor' | 'nodes-editor';

const ViewContext = createContext<ViewType | null>(null);

export const ViewContextProvider = memo((props: PropsWithChildren<{ viewType: ViewType }>) => {
  return <ViewContext.Provider value={props.viewType}>{props.children}</ViewContext.Provider>;
});

ViewContextProvider.displayName = 'ViewContextProvider';

export const useViewContext = () => {
  const context = useContext(ViewContext);
  assert(context !== null, 'useViewContext must be used within a ViewContextProvider');
  return context;
};
