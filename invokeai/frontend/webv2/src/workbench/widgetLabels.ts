import type { TFunction } from 'i18next';

import type { WidgetManifest, WidgetTypeId } from './types';

type WidgetLabelSource = Pick<WidgetManifest, 'id' | 'label'>;

export const resolveWidgetLabel = (manifest: WidgetLabelSource, t: TFunction): string =>
  typeof manifest.label === 'function' ? manifest.label(t) : manifest.label;

export const getWidgetFallbackLabel = (manifest: { id: WidgetTypeId; label: WidgetManifest['label'] }): string =>
  typeof manifest.label === 'string' ? manifest.label : manifest.id;
