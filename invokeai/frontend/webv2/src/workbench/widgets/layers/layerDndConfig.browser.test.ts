import { describe, expect, it } from 'vitest';

import { shouldStartLayerKeyboardDrag } from './layerDndConfig';

const keyEvent = (target: EventTarget | null) => ({
  altKey: false,
  ctrlKey: false,
  metaKey: false,
  shiftKey: false,
  target,
});

/** A layer row containing a button, plus a portalled panel outside it. */
const buildTree = () => {
  const row = document.createElement('div');
  const button = document.createElement('button');
  const input = document.createElement('input');
  row.append(button, input);

  const portal = document.createElement('div');
  const portalTextarea = document.createElement('textarea');
  portal.append(portalTextarea);

  document.body.append(row, portal);
  return { button, input, portal, portalTextarea, row };
};

describe('shouldStartLayerKeyboardDrag — DOM rules', () => {
  it('allows a plain Enter on the row itself', () => {
    const { row } = buildTree();
    expect(shouldStartLayerKeyboardDrag(keyEvent(row), row)).toBe(true);
  });

  it('allows a plain Enter on a non-editable child of the row', () => {
    const { button, row } = buildTree();
    expect(shouldStartLayerKeyboardDrag(keyEvent(button), row)).toBe(true);
  });

  it('refuses Enter typed in the row own rename input', () => {
    const { input, row } = buildTree();
    expect(shouldStartLayerKeyboardDrag(keyEvent(input), row)).toBe(false);
  });

  it('refuses Enter from a PORTALLED textarea that bubbled in through the React tree', () => {
    // The layer-properties popover renders through a Portal, so its DOM node is
    // not inside the row even though React events reach the row.
    const { portalTextarea, row } = buildTree();
    expect(shouldStartLayerKeyboardDrag(keyEvent(portalTextarea), row)).toBe(false);
  });

  it('refuses Enter from any node outside the row, editable or not', () => {
    const { portal, row } = buildTree();
    expect(shouldStartLayerKeyboardDrag(keyEvent(portal), row)).toBe(false);
  });

  it('refuses a modified Enter even on the row itself', () => {
    const { row } = buildTree();
    expect(shouldStartLayerKeyboardDrag({ ...keyEvent(row), ctrlKey: true }, row)).toBe(false);
  });
});
