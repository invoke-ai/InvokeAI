import type { ToolId } from '@workbench/canvas-engine/api';

import { createExternalStore } from '@platform/state/externalStore';

/** Tool strip slots that hold several tools. Storage keys are stored in browsers: never rename them. */
const TOOL_FAMILIES = {
  fill: {
    defaultTool: 'gradient',
    members: ['gradient', 'paintBucket'],
    storageKey: 'invokeai:v7:webv2:fill-family-tool',
  },
  select: { defaultTool: 'marquee', members: ['marquee', 'lasso'], storageKey: 'invokeai:v7:webv2:select-family-tool' },
} as const satisfies Record<string, { defaultTool: ToolId; members: readonly ToolId[]; storageKey: string }>;

export type ToolFamily = keyof typeof TOOL_FAMILIES;
export type ToolFamilyMember<Family extends ToolFamily> = (typeof TOOL_FAMILIES)[Family]['members'][number];

/** The tool as a member of the family, or null when it belongs elsewhere. */
export const familyMemberOf = <Family extends ToolFamily>(
  family: Family,
  tool: unknown
): ToolFamilyMember<Family> | null =>
  (TOOL_FAMILIES[family].members as readonly unknown[]).includes(tool) ? (tool as ToolFamilyMember<Family>) : null;

const readStored = <Family extends ToolFamily>(family: Family): ToolFamilyMember<Family> => {
  const { defaultTool, storageKey } = TOOL_FAMILIES[family];
  try {
    return familyMemberOf(family, window.localStorage.getItem(storageKey)) ?? defaultTool;
  } catch {
    return defaultTool;
  }
};

/** Remember the last member used in each family, so a plain slot click or the family key restores it. */
const store = createExternalStore<{ [Family in ToolFamily]: ToolFamilyMember<Family> }>({
  fill: readStored('fill'),
  select: readStored('select'),
});

export const recordFamilyTool = <Family extends ToolFamily>(family: Family, tool: ToolFamilyMember<Family>): void => {
  const snapshot = store.getSnapshot();
  if (snapshot[family] === tool) {
    return;
  }
  store.setSnapshot({ ...snapshot, [family]: tool });
  try {
    window.localStorage.setItem(TOOL_FAMILIES[family].storageKey, tool);
  } catch {
    // Storage failures are non-fatal; this is a convenience.
  }
};

/** The remembered member, read outside React (hotkeys). */
export const rememberedFamilyTool = <Family extends ToolFamily>(family: Family): ToolFamilyMember<Family> =>
  store.getSnapshot()[family];

export const useRememberedFamilyTool = <Family extends ToolFamily>(family: Family): ToolFamilyMember<Family> =>
  store.useSelector((snapshot) => snapshot[family], Object.is);
