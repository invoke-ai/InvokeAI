import { isGalleryItemDragData } from '@features/gallery/utility';

/**
 * Drop plumbing for the media (image/video) workflow field inputs: a gallery
 * item dragged onto a field's input row sets that field's value.
 */

export type WorkflowMediaKind = 'image' | 'video';

export interface WorkflowMediaDropItem {
  kind: WorkflowMediaKind;
  name: string;
}

export const getWorkflowMediaFieldDropId = (fieldKey: string): string => `workflow-media-field:${fieldKey}`;

/**
 * Resolves a gallery drag payload to the single item a media field can accept,
 * or null. Multi-item drags are rejected outright: a single-value field
 * silently keeping only the first of several dragged items would misread the
 * user's intent.
 */
export const getWorkflowMediaFieldDropItem = (
  activeData: unknown,
  kind: WorkflowMediaKind
): WorkflowMediaDropItem | null => {
  if (!isGalleryItemDragData(activeData) || activeData.items.length !== 1) {
    return null;
  }

  const item = activeData.items[0];

  return item && item.kind === kind ? { kind: item.kind, name: item.name } : null;
};
