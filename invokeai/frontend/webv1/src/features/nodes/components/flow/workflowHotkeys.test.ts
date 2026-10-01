import { describe, expect, it } from 'vitest';

import { isEventTargetWithinElement, isWorkflowHotkeyEnabled, shouldIgnoreWorkflowCopyHotkey } from './workflowHotkeys';

describe('isEventTargetWithinElement', () => {
  it('returns true when the element contains the event target', () => {
    const target = new EventTarget();
    const element = {
      contains: (node: unknown) => node === target,
    };

    expect(isEventTargetWithinElement(target, element as never)).toBe(true);
  });

  it('returns false when the element does not contain the event target', () => {
    const target = new EventTarget();
    const element = {
      contains: () => false,
    };

    expect(isEventTargetWithinElement(target, element as never)).toBe(false);
  });

  it('returns false when the element is missing', () => {
    expect(isEventTargetWithinElement(new EventTarget(), null)).toBe(false);
  });
});

describe('isWorkflowHotkeyEnabled', () => {
  it('enables workflow hotkeys whenever the workflows pane is focused', () => {
    expect(isWorkflowHotkeyEnabled(true)).toBe(true);
  });

  it('disables workflow hotkeys when the workflows pane is not focused', () => {
    expect(isWorkflowHotkeyEnabled(false)).toBe(false);
  });
});

describe('shouldIgnoreWorkflowCopyHotkey', () => {
  const insideNode = new EventTarget() as Node;
  const outsideNode = new EventTarget() as Node;
  const element = {
    contains: (node: Node) => node === insideNode,
  };

  it('returns false when there is no selection', () => {
    expect(shouldIgnoreWorkflowCopyHotkey(null, element)).toBe(false);
  });

  it('returns false for collapsed selections', () => {
    expect(
      shouldIgnoreWorkflowCopyHotkey(
        { isCollapsed: true, toString: () => 'text', anchorNode: outsideNode, focusNode: outsideNode },
        element
      )
    ).toBe(false);
  });

  it('returns false when the selection is inside the editor element', () => {
    expect(
      shouldIgnoreWorkflowCopyHotkey(
        { isCollapsed: false, toString: () => 'text', anchorNode: insideNode, focusNode: insideNode },
        element
      )
    ).toBe(false);
  });

  it('returns true when the selection is outside the editor element', () => {
    expect(
      shouldIgnoreWorkflowCopyHotkey(
        { isCollapsed: false, toString: () => 'text', anchorNode: outsideNode, focusNode: outsideNode },
        element
      )
    ).toBe(true);
  });
});
