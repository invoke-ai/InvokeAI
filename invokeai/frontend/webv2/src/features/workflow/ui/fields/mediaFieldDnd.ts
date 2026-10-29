import { isGalleryItemDragData } from '@features/gallery/utility';

export type WorkflowMediaKind = 'image' | 'video';

export interface WorkflowMediaDropItem {
  kind: WorkflowMediaKind;
  name: string;
}

export const getWorkflowMediaFieldDropId = (fieldKey: string): string => `workflow-media-field:${fieldKey}`;

/** Every dragged gallery item of `kind`; empty when the drag carries anything else. */
export const getWorkflowMediaFieldDropItems = (
  activeData: unknown,
  kind: WorkflowMediaKind
): WorkflowMediaDropItem[] =>
  isGalleryItemDragData(activeData) && activeData.items.every((item) => item.kind === kind)
    ? activeData.items.map((item) => ({ kind, name: item.name }))
    : [];

/** Reject multi-item drags for single-value media fields instead of silently taking the first. */
export const getWorkflowMediaFieldDropItem = (
  activeData: unknown,
  kind: WorkflowMediaKind
): WorkflowMediaDropItem | null => {
  const items = getWorkflowMediaFieldDropItems(activeData, kind);

  return items.length === 1 ? (items[0] ?? null) : null;
};
