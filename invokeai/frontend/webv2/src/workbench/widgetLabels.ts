import type { WidgetRegion } from '@workbench/layoutContracts';
import type { WidgetInstanceRuntimeMeta, WidgetManifest, WidgetTypeId } from '@workbench/widgetContracts';
import type { TFunction } from 'i18next';
import type { LucideIcon } from 'lucide-react';

import { AppWindowIcon, PanelBottomIcon, PanelLeftIcon, PanelRightIcon } from 'lucide-react';

type WidgetLabelSource = Pick<WidgetManifest, 'id' | 'label'>;

export const resolveWidgetLabel = (manifest: WidgetLabelSource, t: TFunction): string =>
  typeof manifest.label === 'function' ? manifest.label(t) : manifest.label;

export const resolveWidgetInstanceLabel = (
  instance: Pick<WidgetInstanceRuntimeMeta, 'title'>,
  manifest: WidgetLabelSource,
  t: TFunction
): string => instance.title ?? resolveWidgetLabel(manifest, t);

/** Names where docking a floating window puts it, e.g. "Dock to right panel". */
export const resolveDockLabel = (returnRegion: WidgetRegion, t: TFunction): string =>
  t('widgets.floating.dockTo', { destination: t(`widgets.floating.destinations.${returnRegion}`) });

/** The icon that goes with {@link resolveDockLabel}, by the region the window returns to. */
export const DOCK_DESTINATION_ICONS: Record<WidgetRegion, LucideIcon> = {
  bottom: PanelBottomIcon,
  center: AppWindowIcon,
  left: PanelLeftIcon,
  right: PanelRightIcon,
};

export const getWidgetFallbackLabel = (manifest: { id: WidgetTypeId; label: WidgetManifest['label'] }): string =>
  typeof manifest.label === 'string' ? manifest.label : manifest.id;
