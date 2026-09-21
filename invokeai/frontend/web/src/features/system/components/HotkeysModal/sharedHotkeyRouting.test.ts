import { describe, expect, it } from 'vitest';

import { getSharedInfoHotkeyTarget } from './sharedHotkeyRouting';

describe('getSharedInfoHotkeyTarget', () => {
  it.each([
    { isGalleryFocused: true, isViewerFocused: false },
    { isGalleryFocused: false, isViewerFocused: true },
  ])('routes the shared hotkey to metadata when Gallery or Viewer owns focus', (focusedRegion) => {
    expect(
      getSharedInfoHotkeyTarget({
        hasMetadataViewerItem: true,
        isViewerPanelActive: true,
        ...focusedRegion,
      })
    ).toBe('metadata');
  });

  it('keeps the shared hotkey on the color picker when Gallery is next to the active Canvas', () => {
    expect(
      getSharedInfoHotkeyTarget({
        hasMetadataViewerItem: true,
        isGalleryFocused: true,
        isViewerPanelActive: false,
        isViewerFocused: false,
      })
    ).toBe('colorPicker');
  });

  it('routes the shared hotkey to the color picker outside Gallery and Viewer', () => {
    expect(
      getSharedInfoHotkeyTarget({
        hasMetadataViewerItem: true,
        isGalleryFocused: false,
        isViewerPanelActive: true,
        isViewerFocused: false,
      })
    ).toBe('colorPicker');
  });

  it.each([
    { isGalleryFocused: true, isViewerFocused: false },
    { isGalleryFocused: false, isViewerFocused: true },
  ])('falls back to the color picker when the focused media panel has no selected item', (focusedRegion) => {
    expect(
      getSharedInfoHotkeyTarget({
        hasMetadataViewerItem: false,
        isViewerPanelActive: true,
        ...focusedRegion,
      })
    ).toBe('colorPicker');
  });
});
