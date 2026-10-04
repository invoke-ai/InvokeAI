type SharedInfoHotkeyTarget = 'colorPicker' | 'metadata';

export const getSharedInfoHotkeyTarget = ({
  hasMetadataViewerItem,
  isGalleryFocused,
  isViewerPanelActive,
  isViewerFocused,
}: {
  hasMetadataViewerItem: boolean;
  isGalleryFocused: boolean;
  isViewerPanelActive: boolean;
  isViewerFocused: boolean;
}): SharedInfoHotkeyTarget => {
  if (hasMetadataViewerItem && isViewerPanelActive && (isGalleryFocused || isViewerFocused)) {
    return 'metadata';
  }

  return 'colorPicker';
};
