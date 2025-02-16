import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext } from 'react';
import { assert } from 'tsafe';

type View = 'view-mode-linear' | 'edit-mode-linear' | 'edit-mode-nodes';

const ViewContext = createContext<View | null>(null);

type Props = PropsWithChildren<{ view: View }>;

export const ViewContextProvider = memo((props: Props) => {
  return <ViewContext.Provider value={props.view}>{props.children}</ViewContext.Provider>;
});

ViewContextProvider.displayName = 'ViewContextProvider';

export const useViewContext = () => {
  const context = useContext(ViewContext);
  assert(context !== null, 'useViewContext must be used within a ViewContextProvider');
  return context;
};
