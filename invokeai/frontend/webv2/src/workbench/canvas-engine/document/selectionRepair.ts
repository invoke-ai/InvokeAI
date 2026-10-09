import type { CanvasStackForests } from '@workbench/canvas-engine/contracts';

import { compileSemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';

import { getDocumentIndex, indexStacks, type CanvasDocumentIndex, type CanvasNodeEntry } from './documentIndex';
import { isGroupNode } from './documentTree';
import { isLeafEditable } from './layerEligibility';
import { isOverlayStack } from './layerStacks';

const nearestInOrder = (
  entries: readonly CanvasNodeEntry[],
  from: number,
  accept: (entry: CanvasNodeEntry) => boolean
): string | null => {
  for (let position = from + 1; position < entries.length; position += 1) {
    if (accept(entries[position]!)) {
      return entries[position]!.node.id;
    }
  }
  for (let position = from - 1; position >= 0; position -= 1) {
    if (accept(entries[position]!)) {
      return entries[position]!.node.id;
    }
  }
  return null;
};

const siblingsOf = (index: CanvasDocumentIndex, entry: CanvasNodeEntry): readonly CanvasNodeEntry[] =>
  index.nodes.filter((candidate) => candidate.stack === entry.stack && candidate.parentId === entry.parentId);

/**
 * The primary selection after `stacks` replaced `previous`. A surviving selection is kept. A removed
 * one moves to its nearest surviving sibling in the previous order (below first, then above), then
 * to its previous parent, then to the nearest surviving node of its stack, then to the nearest
 * surviving node of any stack; without a previous order the top node is selected.
 */
export const repairSelectedLayerId = (
  stacks: CanvasStackForests,
  selectedLayerId: string | null,
  previous?: CanvasStackForests
): string | null => {
  if (selectedLayerId === null) {
    return null;
  }
  const next = indexStacks(stacks);
  if (next.byId.has(selectedLayerId)) {
    return selectedLayerId;
  }
  const survives = (entry: CanvasNodeEntry): boolean => next.byId.has(entry.node.id);
  const before = previous ? indexStacks(previous) : null;
  const entry = before?.byId.get(selectedLayerId);
  if (before && entry) {
    const siblings = siblingsOf(before, entry);
    return (
      nearestInOrder(siblings, siblings.indexOf(entry), survives) ??
      (entry.parentId !== null && next.byId.has(entry.parentId) ? entry.parentId : null) ??
      nearestInOrder(
        before.nodes,
        entry.order,
        (candidate) => candidate.stack === entry.stack && survives(candidate)
      ) ??
      nearestInOrder(before.nodes, entry.order, survives) ??
      next.nodes[0]?.node.id ??
      null
    );
  }
  return next.nodes[0]?.node.id ?? null;
};

/**
 * The primary selection once an accepted result lands as a new raster layer on top of the raster stack. A selected
 * overlay leaf (an inpaint or regional mask, a control layer) stays the editing target while the next stroke can
 * edit it, by the rule painting uses: it and every ancestor enabled and unlocked. Otherwise the result is selected:
 * nothing, a missing node or a group (no paint target) was selected, the overlay leaf would refuse the stroke, or
 * the selection was raster work the result now covers. The accept adds only a raster leaf, so the controller (on the
 * stacks before it) and the rollback (on the stacks after it) see the same entry for the selection and agree.
 */
export const selectionAfterAcceptedResult = (
  stacks: CanvasStackForests,
  selectedLayerId: string | null,
  acceptedLayerId: string
): string => {
  const entry = selectedLayerId === null ? undefined : getDocumentIndex({ stacks }).byId.get(selectedLayerId);
  return entry &&
    isOverlayStack(entry.stack) &&
    !isGroupNode(entry.node) &&
    isLeafEditable(compileSemanticLeaf(entry.node, entry))
    ? entry.node.id
    : acceptedLayerId;
};
