export const isEventTargetWithinElement = (
  target: EventTarget | null,
  element: { contains: (node: Node) => boolean } | null
) => {
  return Boolean(target && element?.contains(target as Node));
};

export const isWorkflowHotkeyEnabled = (isWorkflowsFocused: boolean) => {
  return isWorkflowsFocused;
};

type SelectionLike = {
  isCollapsed: boolean;
  toString(): string;
  anchorNode: Node | null;
  focusNode: Node | null;
};

export const shouldIgnoreWorkflowCopyHotkey = (
  selection: SelectionLike | null | undefined,
  element: { contains: (node: Node) => boolean } | null
) => {
  if (!selection || !element || selection.isCollapsed || selection.toString().length === 0) {
    return false;
  }

  const nodes = [selection.anchorNode, selection.focusNode].filter((node): node is Node => node !== null);

  if (nodes.length === 0) {
    return false;
  }

  return nodes.some((node) => !element.contains(node));
};
