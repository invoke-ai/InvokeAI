import type { GridviewApi } from 'dockview';
import type { Atom } from 'nanostores';
import type { PropsWithChildren } from 'react';
import { createContext, useContext, useMemo } from 'react';

type AutoLayoutContextValue = {
  $api: Atom<GridviewApi | null>;
  toggleLeftPanel: () => void;
  toggleRightPanel: () => void;
};

const AutoLayoutContext = createContext<AutoLayoutContextValue | null>(null);

export const AutoLayoutProvider = (props: PropsWithChildren<AutoLayoutContextValue>) => {
  const value = useMemo<AutoLayoutContextValue>(
    () => ({
      $api: props.$api,
      toggleLeftPanel: props.toggleLeftPanel,
      toggleRightPanel: props.toggleRightPanel,
    }),
    [props.$api, props.toggleLeftPanel, props.toggleRightPanel]
  );
  return <AutoLayoutContext.Provider value={value}>{props.children}</AutoLayoutContext.Provider>;
};

export const useAutoLayoutContext = () => {
  const value = useContext(AutoLayoutContext);
  if (!value) {
    throw new Error('useAutoLayoutContext must be used within an AutoLayoutProvider');
  }
  return value;
};
