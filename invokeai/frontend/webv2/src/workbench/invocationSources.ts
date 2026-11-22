import type { InvocationSourceId } from './invocationContracts';
import type { WidgetTypeId } from './widgetContracts';

import { firstPartyWidgetManifests } from './widgets/manifests';

export interface InvocationSourceMeta {
  id: InvocationSourceId;
  label: string;
  available: boolean;
}

/** In the order the source pickers list them. */
export const invocationSources: InvocationSourceMeta[] = [
  { id: 'generate', label: 'Generate', available: true },
  { id: 'canvas', label: 'Canvas', available: true },
  { id: 'upscale', label: 'Upscale', available: true },
  { id: 'video', label: 'Video', available: true },
  { id: 'workflow', label: 'Workflow', available: true },
];

export const isInvocationSourceAvailable = (id: InvocationSourceId): boolean =>
  invocationSources.some((source) => source.id === id && source.available);

interface SourceWidgets {
  bySource: ReadonlyMap<InvocationSourceId, WidgetTypeId>;
  byType: ReadonlyMap<WidgetTypeId, InvocationSourceId>;
}

// Read on first use, never at import, so an import cycle through the manifests cannot meet them uninitialized.
let sourceWidgets: SourceWidgets | undefined;

/** Each manifest's `graphBearing.sourceId`, the one widget-to-source mapping; manifest order picks a source's widget. */
const getSourceWidgets = (): SourceWidgets => {
  if (!sourceWidgets) {
    const bySource = new Map<InvocationSourceId, WidgetTypeId>();
    const byType = new Map<WidgetTypeId, InvocationSourceId>();

    for (const manifest of firstPartyWidgetManifests) {
      const sourceId = manifest.graphBearing?.sourceId;

      if (sourceId) {
        byType.set(manifest.id, sourceId);

        if (!bySource.has(sourceId)) {
          bySource.set(sourceId, manifest.id);
        }
      }
    }

    sourceWidgets = { bySource, byType };
  }

  return sourceWidgets;
};

export const getSourceIdForWidgetTypeId = (typeId: WidgetTypeId): InvocationSourceId | null =>
  getSourceWidgets().byType.get(typeId) ?? null;

/** The widget that runs a source, or null when no manifest claims it. */
export const findWidgetTypeIdForSourceId = (sourceId: InvocationSourceId): WidgetTypeId | null =>
  getSourceWidgets().bySource.get(sourceId) ?? null;

/** The widget that runs a source; the source's own id when no manifest claims it. */
export const getWidgetTypeIdForSourceId = (sourceId: InvocationSourceId): WidgetTypeId =>
  findWidgetTypeIdForSourceId(sourceId) ?? sourceId;

/** The source a widget type runs while that source is available, else null. */
export const getAvailableSourceIdForWidgetTypeId = (typeId: WidgetTypeId): InvocationSourceId | null => {
  const sourceId = getSourceIdForWidgetTypeId(typeId);

  return sourceId && isInvocationSourceAvailable(sourceId) ? sourceId : null;
};
