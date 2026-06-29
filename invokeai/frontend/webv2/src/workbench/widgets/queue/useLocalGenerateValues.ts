import type { GenerateWidgetValues } from '@workbench/generation/types';

import { parseQueueItemOrigin } from '@workbench/backend/events';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';

/**
 * The full Generate settings snapshot for a server queue item, if this client
 * submitted it. webv2 stamps the local submission id into each item's `origin`,
 * so we decode it and look the snapshot up in local state. Present → "Use again"
 * can recall the exact settings (model, steps, LoRAs, …); absent (item from
 * another client, or a cleared snapshot) → only field_values are recoverable.
 */
export const useLocalGenerateValues = (origin?: string | null): GenerateWidgetValues | null =>
  useWorkbenchSelector((snapshot) => {
    const localId = parseQueueItemOrigin(origin);

    if (!localId) {
      return null;
    }

    for (const project of snapshot.state.projects) {
      const localItem = project.queue.items.find((queueItem) => queueItem.id === localId);

      if (localItem) {
        const generate = (localItem.snapshot.widgetStates as Record<string, { values?: unknown }>).generate;

        return (generate?.values as GenerateWidgetValues | undefined) ?? null;
      }
    }

    return null;
  });
