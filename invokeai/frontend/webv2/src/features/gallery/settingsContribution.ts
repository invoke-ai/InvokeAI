import type { SettingDefinition } from '@platform/ui/settings/contracts';

/**
 * The Gallery's settings metadata. Each is layered per project (pin, the user's default, built-in), so the Workbench
 * binds them: inside a project they pin, in the Launchpad they edit the default. `imageSize` shows 100 minus the stored
 * `imageDensityPercent`; every other id is its `GallerySettings` key.
 */
export const gallerySettingsFields: readonly SettingDefinition[] = [
  {
    id: 'imageSize',
    kind: 'slider',
    label: (t) => t('widgets.gallery.imageSize'),
    group: (t) => t('widgets.gallery.settingsGridSize'),
    keywords: 'density thumbnails zoom',
    min: 0,
    max: 100,
    step: 1,
    scope: 'project',
  },
  {
    id: 'thumbnailFit',
    kind: 'select',
    label: (t) => t('widgets.gallery.thumbnailShape'),
    group: (t) => t('widgets.gallery.settingsThumbnails'),
    keywords: 'square aspect shape crop fit',
    options: [
      { value: 'square', label: (t) => t('widgets.gallery.thumbnailFitSquare') },
      { value: 'aspect', label: (t) => t('widgets.gallery.thumbnailFitAspect') },
    ],
    scope: 'project',
  },
  {
    id: 'showImageDimensions',
    kind: 'boolean',
    label: (t) => t('widgets.gallery.alwaysShowDimensions'),
    group: (t) => t('widgets.gallery.settingsThumbnails'),
    keywords: 'width height pixels',
    scope: 'project',
  },
  {
    id: 'showPendingItems',
    kind: 'boolean',
    label: (t) => t('widgets.gallery.showPendingItems'),
    group: (t) => t('widgets.gallery.settingsThumbnails'),
    keywords: 'queue progress placeholders',
    scope: 'project',
  },
  {
    id: 'middleClickOpensInNewTab',
    kind: 'boolean',
    label: (t) => t('widgets.gallery.middleClickOpensInNewTab'),
    group: (t) => t('widgets.gallery.settingsThumbnails'),
    keywords: 'mouse middle click wheel button open new tab',
    scope: 'project',
  },
  {
    id: 'paginationMode',
    kind: 'select',
    label: (t) => t('widgets.gallery.paginationMode'),
    group: (t) => t('widgets.gallery.pagination'),
    keywords: 'infinite scroll pages pagination',
    options: [
      { value: 'infinite', label: (t) => t('widgets.gallery.infinite') },
      { value: 'paginated', label: (t) => t('common.pages') },
    ],
    scope: 'project',
  },
  {
    id: 'imageOrderDir',
    kind: 'select',
    label: (t) => t('widgets.gallery.imageSort'),
    group: (t) => t('widgets.gallery.pagination'),
    options: [
      { value: 'DESC', label: (t) => t('widgets.gallery.newest') },
      { value: 'ASC', label: (t) => t('widgets.gallery.oldest') },
    ],
    scope: 'project',
  },
  {
    id: 'followSelectedBoard',
    kind: 'boolean',
    label: (t) => t('widgets.gallery.followSelectedBoard'),
    description: (t) => t('widgets.gallery.followSelectedBoardDescription'),
    group: (t) => t('widgets.gallery.destination'),
    keywords: 'auto add board destination results workflow',
    scope: 'project',
  },
  {
    id: 'showDateBoards',
    kind: 'boolean',
    label: (t) => t('widgets.gallery.dateBoards'),
    group: (t) => t('widgets.gallery.boardVisibility'),
    scope: 'project',
  },
  {
    id: 'showArchivedBoards',
    kind: 'boolean',
    label: (t) => t('widgets.gallery.archivedBoards'),
    group: (t) => t('widgets.gallery.boardVisibility'),
    scope: 'project',
  },
  {
    id: 'showOtherProjectBoards',
    kind: 'boolean',
    label: (t) => t('widgets.gallery.otherProjectBoards'),
    group: (t) => t('widgets.gallery.boardVisibility'),
    scope: 'project',
  },
  {
    id: 'boardOrderBy',
    kind: 'select',
    label: (t) => t('widgets.gallery.sortBoardsBy'),
    group: (t) => t('widgets.gallery.filterAndSortBoards'),
    options: [
      { value: 'created_at', label: (t) => t('common.created') },
      { value: 'board_name', label: (t) => t('common.name') },
    ],
    scope: 'project',
  },
  {
    id: 'boardOrderDir',
    kind: 'select',
    label: (t) => t('widgets.gallery.boardOrder', { defaultValue: 'Board order' }),
    group: (t) => t('widgets.gallery.filterAndSortBoards'),
    options: [
      { value: 'DESC', label: (t) => t('widgets.gallery.sortBoardsDescending') },
      { value: 'ASC', label: (t) => t('widgets.gallery.sortBoardsAscending') },
    ],
    scope: 'project',
  },
];

/** The fields the Gallery's gear popover shows, in order. */
export const galleryQuickSettings: readonly string[] = [
  'imageSize',
  'thumbnailFit',
  'showImageDimensions',
  'showPendingItems',
  'paginationMode',
];
