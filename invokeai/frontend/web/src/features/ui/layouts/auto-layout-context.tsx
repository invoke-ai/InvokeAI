import type { FocusRegionName } from 'common/hooks/focus';
import type { IDockviewPanelProps, IGridviewPanelProps } from 'dockview';
import type { TabName } from 'features/ui/store/uiTypes';
import type { FunctionComponent, PropsWithChildren, RefObject } from 'react';
import { createContext, memo, useContext, useMemo } from 'react';

import { AutoLayoutPanelContainer } from './AutoLayoutPanelContainer';

type AutoLayoutContextValue = {
  tab: TabName;
  rootRef: RefObject<HTMLDivElement>;
};

const AutoLayoutContext = createContext<AutoLayoutContextValue | null>(null);

export const AutoLayoutProvider = (props: PropsWithChildren<AutoLayoutContextValue>) => {
  const { tab, rootRef, children } = props;

  const value = useMemo<AutoLayoutContextValue>(() => ({ tab, rootRef }), [tab, rootRef]);
  return <AutoLayoutContext.Provider value={value}>{children}</AutoLayoutContext.Provider>;
};

export const useAutoLayoutContext = () => {
  const value = useContext(AutoLayoutContext);
  if (!value) {
    throw new Error('useAutoLayoutContext must be used within an AutoLayoutProvider');
  }
  return value;
};

export const useAutoLayoutContextSafe = () => {
  const value = useContext(AutoLayoutContext);
  return value;
};

export type PanelParameters = {
  tab: TabName;
  focusRegion: FocusRegionName;
};

export type AutoLayoutGridviewComponents = Record<string, FunctionComponent<IGridviewPanelProps<PanelParameters>>>;
export type AutoLayoutDockviewComponents = Record<string, FunctionComponent<IDockviewPanelProps<PanelParameters>>>;
export type RootLayoutGridviewComponents = Record<string, FunctionComponent<IGridviewPanelProps<PanelParameters>>>;
export type PanelProps = IDockviewPanelProps<PanelParameters> | IGridviewPanelProps<PanelParameters>;

export const withPanelContainer = (Component: FunctionComponent) =>
  memo((props: PanelProps) => {
    return (
      <AutoLayoutPanelContainer {...props}>
        <Component />
      </AutoLayoutPanelContainer>
    );
  });
