import type { InvocationSourceId } from './invocationContracts';
import type { WidgetTypeId } from './widgetContracts';

export interface InvocationSourceMeta {
  id: InvocationSourceId;
  label: string;
  available: boolean;
}

export const invocationSources: InvocationSourceMeta[] = [
  { id: 'generate', label: 'Generate', available: true },
  { id: 'workflow', label: 'Workflow', available: true },
  { id: 'upscale', label: 'Upscale', available: true },
  { id: 'video', label: 'Video', available: true },
  { id: 'canvas', label: 'Canvas', available: true },
];

export const isInvocationSourceAvailable = (id: InvocationSourceId): boolean =>
  invocationSources.some((source) => source.id === id && source.available);

export const graphWidgetTypeIds = ['generate', 'canvas', 'upscale', 'video', 'workflow'] as const;

export type GraphWidgetTypeId = (typeof graphWidgetTypeIds)[number];

export const sourceIdsByWidgetTypeId: Record<GraphWidgetTypeId, InvocationSourceId> = {
  canvas: 'canvas',
  generate: 'generate',
  upscale: 'upscale',
  video: 'video',
  workflow: 'workflow',
};

const widgetTypeIdsBySourceId: Record<InvocationSourceId, GraphWidgetTypeId> = {
  canvas: 'canvas',
  generate: 'generate',
  upscale: 'upscale',
  video: 'video',
  workflow: 'workflow',
};

export const isGraphWidgetTypeId = (typeId: WidgetTypeId): typeId is GraphWidgetTypeId =>
  (graphWidgetTypeIds as readonly string[]).includes(typeId);

export const getSourceIdForWidgetTypeId = (typeId: WidgetTypeId): InvocationSourceId | null =>
  isGraphWidgetTypeId(typeId) ? sourceIdsByWidgetTypeId[typeId] : null;

export const getWidgetTypeIdForSourceId = (sourceId: InvocationSourceId): GraphWidgetTypeId =>
  widgetTypeIdsBySourceId[sourceId];

/** The source a widget type runs while that source is available, else null. */
export const getAvailableSourceIdForWidgetTypeId = (typeId: WidgetTypeId): InvocationSourceId | null => {
  const sourceId = getSourceIdForWidgetTypeId(typeId);

  return sourceId && isInvocationSourceAvailable(sourceId) ? sourceId : null;
};
