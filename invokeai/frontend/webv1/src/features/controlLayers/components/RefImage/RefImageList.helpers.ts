import type { Edge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/closest-edge';
import { reorderWithEdge } from '@atlaskit/pragmatic-drag-and-drop-hitbox/util/reorder-with-edge';

type GetReorderedRefImageIdsParams = {
  ids: string[];
  sourceId: string;
  targetId: string;
  closestEdgeOfTarget: Edge | null;
};

/**
 * Computes the reordered id list for a horizontal ref-image drag-and-drop.
 *
 * Returns `null` for any drop that should be a no-op:
 * - Source or target id not present in `ids`.
 * - Source and target are the same item.
 * - The item is already on the side of the target indicated by `closestEdgeOfTarget`.
 *
 * Notes on `closestEdgeOfTarget = null`: pragmatic-dnd's `extractClosestEdge` may return `null`
 * when the hitbox util cannot determine a side. `reorderWithEdge` then treats the destination
 * as `indexOfTarget` (i.e. moves the source onto the target's slot). The no-op short-circuit
 * cannot fire in this case (`edgeIndexDelta` is 0, but `indexOfSource === indexOfTarget` was
 * already rejected), so the move is forwarded to the util.
 */
export const getReorderedRefImageIds = ({
  ids,
  sourceId,
  targetId,
  closestEdgeOfTarget,
}: GetReorderedRefImageIdsParams): string[] | null => {
  const indexOfSource = ids.indexOf(sourceId);
  const indexOfTarget = ids.indexOf(targetId);

  if (indexOfTarget < 0 || indexOfSource < 0) {
    return null;
  }

  if (indexOfSource === indexOfTarget) {
    return null;
  }

  let edgeIndexDelta = 0;
  if (closestEdgeOfTarget === 'right') {
    edgeIndexDelta = 1;
  } else if (closestEdgeOfTarget === 'left') {
    edgeIndexDelta = -1;
  }

  if (indexOfSource === indexOfTarget + edgeIndexDelta) {
    return null;
  }

  return reorderWithEdge({
    list: ids,
    startIndex: indexOfSource,
    indexOfTarget,
    closestEdgeOfTarget,
    axis: 'horizontal',
  });
};
