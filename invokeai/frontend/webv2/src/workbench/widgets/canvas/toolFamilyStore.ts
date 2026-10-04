import type { ToolId } from '@workbench/canvas-engine/api';

import { createExternalStore } from '@platform/state/externalStore';

const STORAGE_KEY = 'invokeai:v7:webv2:select-family-tool';

export type SelectFamilyTool = Extract<ToolId, 'marquee' | 'lasso'>;

const isSelectFamilyTool = (value: unknown): value is SelectFamilyTool => value === 'marquee' || value === 'lasso';

const readStored = (): SelectFamilyTool => {
  if (typeof window === 'undefined') {
    return 'marquee';
  }
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return isSelectFamilyTool(stored) ? stored : 'marquee';
  } catch {
    return 'marquee';
  }
};

/** Remember the user's last selection subtool so a plain family-slot click restores it. */
const store = createExternalStore<{ tool: SelectFamilyTool }>({ tool: readStored() });

export const recordSelectFamilyTool = (tool: SelectFamilyTool): void => {
  if (store.getSnapshot().tool === tool) {
    return;
  }
  store.setSnapshot({ tool });
  try {
    window.localStorage.setItem(STORAGE_KEY, tool);
  } catch {
    // Storage failures are non-fatal; this is a convenience.
  }
};

export const useSelectFamilyTool = (): SelectFamilyTool => store.useSelector((snapshot) => snapshot.tool, Object.is);
