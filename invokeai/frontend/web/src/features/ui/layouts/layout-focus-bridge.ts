import { type FocusRegionName, setFocusedRegion } from 'common/hooks/focus';
import type { DockviewIDisposable, IDockviewPanel, IGridviewPanel } from 'dockview';

export const registerFocusListener = (
  panel: IGridviewPanel | IDockviewPanel,
  region: FocusRegionName
): DockviewIDisposable => {
  return panel.api.onDidActiveChange(({ isActive }) => {
    if (isActive) {
      setFocusedRegion(region);
    }
  });
};
