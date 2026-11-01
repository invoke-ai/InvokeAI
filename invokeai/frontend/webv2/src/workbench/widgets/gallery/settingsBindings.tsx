import type { GallerySettings } from '@features/gallery/contracts';
import type { SettingFieldProps, SettingLayer } from '@platform/ui/settings/contracts';

import { getGallerySettings } from '@features/gallery/contracts';
import { SettingControl } from '@platform/ui/settings/SettingControl';
import { useLayeredSetting } from '@workbench/settings/useLayeredSetting';
import { useOptionalWorkbenchCommands, useOptionalWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useCallback, useMemo } from 'react';

const EMPTY = {};

/** Settings that reorder or repaginate the grid, so a paginated gallery starts again at its first page. */
const RESETS_PAGE = new Set(['imageOrderDir', 'paginationMode']);

const BOOLEAN_FIELDS = new Set([
  'showImageDimensions',
  'showPendingItems',
  'middleClickOpensInNewTab',
  'followSelectedBoard',
  'showDateBoards',
  'showArchivedBoards',
  'showOtherProjectBoards',
]);

/** Whether `value` is one the Gallery accepts for this field; anything else is dropped. */
const accepts = (fieldId: string, value: boolean | number | string): boolean => {
  switch (fieldId) {
    case 'imageSize':
      return typeof value === 'number';
    case 'thumbnailFit':
      return value === 'square' || value === 'aspect';
    case 'paginationMode':
      return value === 'infinite' || value === 'paginated';
    case 'boardOrderBy':
      return value === 'created_at' || value === 'board_name';
    case 'boardOrderDir':
    case 'imageOrderDir':
      return value === 'ASC' || value === 'DESC';
    default:
      return BOOLEAN_FIELDS.has(fieldId) && typeof value === 'boolean';
  }
};

/** The Gallery's settings, layered per project; the size slider shows 100 minus the stored density. */
export const Field = ({ field, surface, target }: SettingFieldProps) => {
  const isImageSize = field.id === 'imageSize';
  const key = isImageSize ? 'imageDensityPercent' : field.id;
  const read = useMemo(
    () =>
      isImageSize
        ? (values: Readonly<Record<string, unknown>>) => 100 - getGallerySettings(values).imageDensityPercent
        : (values: Readonly<Record<string, unknown>>) => {
            const value = getGallerySettings(values)[field.id as keyof GallerySettings];
            return typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string' ? value : null;
          },
    [field.id, isImageSize]
  );
  const { disabled, layer, set, value } = useLayeredSetting('gallery', key, target, read, read(EMPTY), field);
  const commands = useOptionalWorkbenchCommands();
  const queries = useOptionalWorkbenchQueries();
  const projectId = target?.projectId;
  const resetPage = useCallback(() => {
    if (projectId && RESETS_PAGE.has(field.id) && queries?.isActiveProject(projectId)) {
      commands?.gallery.setPage(0, projectId);
    }
  }, [commands, field.id, projectId, queries]);
  const onChange = useCallback(
    (next: boolean | number | string) => {
      if (!accepts(field.id, next)) {
        return;
      }
      set(isImageSize && typeof next === 'number' ? 100 - next : next);
      resetPage();
    },
    [field.id, isImageSize, resetPage, set]
  );
  const galleryLayer = useMemo<SettingLayer | undefined>(
    () =>
      layer?.state === 'pinned' && RESETS_PAGE.has(field.id)
        ? {
            ...layer,
            onFollowDefault: () => {
              layer.onFollowDefault();
              resetPage();
            },
          }
        : layer,
    [field.id, layer, resetPage]
  );

  if (value === null) {
    return null;
  }
  return (
    <SettingControl
      layer={galleryLayer}
      disabled={disabled}
      field={field}
      surface={surface}
      value={value}
      onChange={onChange}
    />
  );
};
