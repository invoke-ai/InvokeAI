import type { SettingFieldProps } from '@platform/ui/settings/contracts';

import { DEFAULT_GALLERY_SETTINGS, GALLERY_AUTO_ADD_FOLLOW, getGallerySettings } from '@features/gallery/core/settings';
import { SettingControl } from '@platform/ui/settings/SettingControl';
import { useCallback } from 'react';

import { getGalleryDestinationBoardId } from './galleryStateView';
import { useGalleryUi } from './GalleryUiContext';

export const GallerySettingField = ({ field, surface, target }: SettingFieldProps) => {
  const { gallery, galleryValues, projectId } = useGalleryUi();
  const settings = getGallerySettings(galleryValues);
  const disabled = target?.projectId !== undefined && target.projectId !== projectId;
  const handleChange = useCallback(
    (value: boolean | number | string) => {
      if (disabled) {
        return;
      }

      if (field.id === 'followSelectedBoard' && typeof value === 'boolean') {
        // Leaving follow pins today's destination, so nothing moves until another board is chosen.
        gallery.updateSettings({
          autoAddBoardId: value ? GALLERY_AUTO_ADD_FOLLOW : (getGalleryDestinationBoardId(galleryValues) ?? 'none'),
        });
      } else if (field.id === 'imageSize' && typeof value === 'number') {
        gallery.updateSettings({ imageDensityPercent: 100 - value });
      } else if (field.id === 'thumbnailFit' && (value === 'square' || value === 'aspect')) {
        gallery.updateSettings({ thumbnailFit: value });
      } else if (field.id === 'paginationMode' && (value === 'infinite' || value === 'paginated')) {
        gallery.updateSettings({ paginationMode: value });
      } else if (field.id === 'boardOrderBy' && (value === 'created_at' || value === 'board_name')) {
        gallery.updateSettings({ boardOrderBy: value });
      } else if (
        (field.id === 'boardOrderDir' || field.id === 'imageOrderDir') &&
        (value === 'ASC' || value === 'DESC')
      ) {
        gallery.updateSettings({ [field.id]: value });
      } else if (
        (field.id === 'showImageDimensions' ||
          field.id === 'showPendingItems' ||
          field.id === 'showDateBoards' ||
          field.id === 'showArchivedBoards' ||
          field.id === 'showOtherProjectBoards') &&
        typeof value === 'boolean'
      ) {
        gallery.updateSettings({ [field.id]: value });
      }
    },
    [disabled, field.id, gallery, galleryValues]
  );
  const value =
    field.id === 'followSelectedBoard'
      ? settings.autoAddBoardId === GALLERY_AUTO_ADD_FOLLOW
      : field.id === 'imageSize'
        ? 100 - settings.imageDensityPercent
        : settings[field.id as keyof typeof settings];

  if (typeof value !== 'boolean' && typeof value !== 'number' && typeof value !== 'string') {
    return null;
  }

  const defaultValue =
    field.id === 'followSelectedBoard'
      ? true
      : field.id === 'imageSize'
        ? 100 - DEFAULT_GALLERY_SETTINGS.imageDensityPercent
        : DEFAULT_GALLERY_SETTINGS[field.id as keyof typeof DEFAULT_GALLERY_SETTINGS];
  return (
    <SettingControl
      isModified={value !== defaultValue}
      disabled={disabled}
      field={field}
      surface={surface}
      value={value}
      onChange={handleChange}
    />
  );
};
