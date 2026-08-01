import type { LucideIcon } from 'lucide-react';

import {
  ApertureIcon,
  BlocksIcon,
  BookmarkIcon,
  BoxIcon,
  BrushIcon,
  CameraIcon,
  Columns2Icon,
  CompassIcon,
  CropIcon,
  FilmIcon,
  FlagIcon,
  FlaskConicalIcon,
  FrameIcon,
  HeartIcon,
  ImageIcon,
  ImagesIcon,
  LayersIcon,
  LayoutDashboardIcon,
  LayoutGridIcon,
  LayoutPanelLeftIcon,
  LayoutPanelTopIcon,
  PaletteIcon,
  PanelsTopLeftIcon,
  PenToolIcon,
  RocketIcon,
  Rows2Icon,
  ShapesIcon,
  SlidersHorizontalIcon,
  SparklesIcon,
  SquareStackIcon,
  StarIcon,
  TypeIcon,
  WandSparklesIcon,
  WorkflowIcon,
  ZapIcon,
} from 'lucide-react';

/**
 * The icons a custom layout preset can be given.
 *
 * A curated set rather than all of Lucide: the picker is a glance-and-choose
 * grid, and a thousand icons would make it a search problem. These are chosen
 * to cover how people actually name a workspace — by its arrangement, by the
 * craft it supports, or by a plain marker when neither applies.
 *
 * Ids are persisted with the preset, so they are stable strings and must not be
 * renamed. Dropping an icon from this list is safe: unknown ids fall back.
 */
export interface LayoutPresetIconOption {
  id: string;
  icon: LucideIcon;
  label: string;
}

export interface LayoutPresetIconGroup {
  label: string;
  options: LayoutPresetIconOption[];
}

const option = (id: string, icon: LucideIcon, label: string): LayoutPresetIconOption => ({ icon, id, label });

export const layoutPresetIconGroups: LayoutPresetIconGroup[] = [
  {
    label: 'Arrangement',
    options: [
      option('layout-grid', LayoutGridIcon, 'Grid'),
      option('layout-dashboard', LayoutDashboardIcon, 'Dashboard'),
      option('layout-panel-left', LayoutPanelLeftIcon, 'Left panel'),
      option('layout-panel-top', LayoutPanelTopIcon, 'Top panel'),
      option('columns', Columns2Icon, 'Columns'),
      option('rows', Rows2Icon, 'Rows'),
      option('panels', PanelsTopLeftIcon, 'Panels'),
      option('square-stack', SquareStackIcon, 'Stack'),
    ],
  },
  {
    label: 'Craft',
    options: [
      option('type', TypeIcon, 'Text'),
      option('brush', BrushIcon, 'Brush'),
      option('pen-tool', PenToolIcon, 'Pen'),
      option('palette', PaletteIcon, 'Palette'),
      option('layers', LayersIcon, 'Layers'),
      option('crop', CropIcon, 'Crop'),
      option('shapes', ShapesIcon, 'Shapes'),
      option('sliders', SlidersHorizontalIcon, 'Settings'),
    ],
  },
  {
    label: 'Subject',
    options: [
      option('image', ImageIcon, 'Image'),
      option('images', ImagesIcon, 'Gallery'),
      option('camera', CameraIcon, 'Camera'),
      option('film', FilmIcon, 'Film'),
      option('frame', FrameIcon, 'Frame'),
      option('aperture', ApertureIcon, 'Aperture'),
      option('workflow', WorkflowIcon, 'Workflow'),
      option('blocks', BlocksIcon, 'Nodes'),
    ],
  },
  {
    label: 'Marker',
    options: [
      option('sparkles', SparklesIcon, 'Sparkles'),
      option('wand', WandSparklesIcon, 'Wand'),
      option('zap', ZapIcon, 'Fast'),
      option('rocket', RocketIcon, 'Rocket'),
      option('flask', FlaskConicalIcon, 'Experiment'),
      option('compass', CompassIcon, 'Explore'),
      option('star', StarIcon, 'Star'),
      option('heart', HeartIcon, 'Favourite'),
      option('bookmark', BookmarkIcon, 'Bookmark'),
      option('flag', FlagIcon, 'Flag'),
      option('box', BoxIcon, 'Box'),
    ],
  },
];

const iconsById = new Map(
  layoutPresetIconGroups.flatMap((group) => group.options.map((entry) => [entry.id, entry.icon]))
);

export const DEFAULT_LAYOUT_PRESET_ICON_ID = 'layout-grid';

/** Unknown or missing ids fall back, so a retired icon never breaks a saved preset. */
export const resolveLayoutPresetIcon = (iconId: string | undefined): LucideIcon =>
  (iconId ? iconsById.get(iconId) : undefined) ?? LayoutGridIcon;

export const isLayoutPresetIconId = (iconId: unknown): iconId is string =>
  typeof iconId === 'string' && iconsById.has(iconId);
