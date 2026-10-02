import type { WidgetInstanceRuntimeMeta, WidgetManifest, WidgetTypeId } from '@workbench/widgetContracts';
import type { TFunction } from 'i18next';

type WidgetLabelSource = Pick<WidgetManifest, 'id' | 'label'>;

export const resolveWidgetLabel = (manifest: WidgetLabelSource, t: TFunction): string =>
  typeof manifest.label === 'function' ? manifest.label(t) : manifest.label;

export const resolveWidgetInstanceLabel = (
  instance: Pick<WidgetInstanceRuntimeMeta, 'title'>,
  manifest: WidgetLabelSource,
  t: TFunction
): string => instance.title ?? resolveWidgetLabel(manifest, t);

export const getWidgetFallbackLabel = (manifest: { id: WidgetTypeId; label: WidgetManifest['label'] }): string =>
  typeof manifest.label === 'string' ? manifest.label : manifest.id;
