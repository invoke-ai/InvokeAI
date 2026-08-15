import type { WorkflowForm } from '@features/workflow/contracts';

/**
 * Pure dnd-kit plumbing for the form builder: droppable id encoding/decoding
 * and the drop-target -> reparent resolution the old native-DnD `onDrop`
 * handlers computed inline. Kept side-effect free so it can be unit tested
 * without mounting the builder; `FormBuilderTab` only wires this to
 * `editGraph({ type: 'moveFormElementTo', ... })` at `onDragEnd`.
 *
 * `WorkflowForm` elements each carry their own `parentId` (maintained by
 * `moveFormElementTo` in `core/document.ts`), so parent lookup is a direct
 * read rather than a scan over every container — mirrors how `getFormChildren`
 * resolves the root the same as any other container (`form.elements[id]`,
 * `type === 'container'`).
 */

export type FormDropTarget =
  | { kind: 'edge'; elementId: string; edge: 'above' | 'below' }
  | { kind: 'into'; containerId: string };

export const formEdgeDroppableId = (elementId: string): string => `edge:${elementId}`;
export const formIntoDroppableId = (containerId: string): string => `into:${containerId}`;

export const parseFormDroppableId = (
  id: string
): { kind: 'edge'; elementId: string } | { kind: 'into'; containerId: string } | null => {
  if (id.startsWith('edge:')) {
    return { elementId: id.slice(5), kind: 'edge' };
  }
  if (id.startsWith('into:')) {
    return { containerId: id.slice(5), kind: 'into' };
  }
  return null;
};

/** The id of `elementId`'s container, or null for the root element (which has no parent). */
export const findFormParentId = (form: WorkflowForm, elementId: string): string | null =>
  form.elements[elementId]?.parentId ?? null;

/** True when `candidateId` is `ancestorId` itself or nested anywhere under it. */
export const isFormDescendantOrSelf = (form: WorkflowForm, ancestorId: string, candidateId: string): boolean => {
  if (ancestorId === candidateId) {
    return true;
  }
  const ancestor = form.elements[ancestorId];
  if (ancestor?.type !== 'container') {
    return false;
  }
  return ancestor.data.children.some((childId) => isFormDescendantOrSelf(form, childId, candidateId));
};

/** Above/below split at the hovered card's midline. `referenceY` is the pointer when available, else the dragged card's center. */
export const getFormDropEdge = (referenceY: number, overTop: number, overHeight: number): 'above' | 'below' =>
  referenceY < overTop + overHeight / 2 ? 'above' : 'below';

/**
 * The container a droppable would actually land the dragged element inside:
 * `into:X` lands inside `X` itself; `edge:X` lands inside `X`'s own parent
 * (it resolves to a sibling reorder around `X`). Unifying both kinds onto
 * this one axis is what makes the tiebreak below a single "deepest wins"
 * comparison instead of two separate rules.
 */
const formDropLandingContainerId = (form: WorkflowForm, droppableId: string): string | null => {
  const parsed = parseFormDroppableId(droppableId);

  if (!parsed) {
    return null;
  }

  return parsed.kind === 'into' ? parsed.containerId : findFormParentId(form, parsed.elementId);
};

/** How many containers separate `containerId` from the form root — 0 for the root itself, +1 per level of nesting. */
const formContainerDepth = (form: WorkflowForm, containerId: string): number => {
  let depth = 0;
  let currentId = containerId;

  while (currentId !== form.rootElementId) {
    const parentId = findFormParentId(form, currentId);

    if (parentId === null) {
      break;
    }
    currentId = parentId;
    depth += 1;
  }

  return depth;
};

/**
 * Innermost-wins tiebreak for overlapping droppables — restores the native-DnD
 * `stopPropagation` semantics the port lost. A container's own `edge:X` and
 * `into:X` are the same ambiguity dnd-kit's rect-based collision can't avoid:
 * `into:X` is DOM-nested inside `edge:X`'s box, so hovering its dropzone
 * always matches both. Comparing by landing container resolves it the same
 * way as any other nesting — the deeper one wins, so `into:X` (lands inside
 * `X`) beats `edge:X` (lands inside `X`'s parent).
 *
 * The rule, precisely: every candidate's landing container has a *depth*
 * (its distance from the form root); the candidate with the greatest depth
 * wins outright, and among candidates tied on depth, an edge wins over an
 * into (a specific insertion point next to a sibling is a more precise
 * intent than "append to this container"). Depth is computed independently
 * per candidate — not by comparing each new candidate only against the
 * current winner — so the result doesn't depend on the order `collisions`
 * arrives in: two incomparable siblings (say containers `C` and `B`, neither
 * nested under the other) plus a deeper `D` nested under `B` must still pick
 * `D`, even if `C` happened to be evaluated first.
 */
export const pickInnermostFormCollision = (collisions: { id: string }[], form: WorkflowForm): string | null => {
  if (collisions.length === 0) {
    return null;
  }

  let winner: string | null = null;
  let winnerDepth = -1;
  let winnerIsEdge = false;

  for (const collision of collisions) {
    const id = String(collision.id);
    const parsed = parseFormDroppableId(id);

    if (!parsed) {
      continue;
    }

    const landingId = formDropLandingContainerId(form, id);

    if (landingId === null) {
      continue;
    }

    const isEdge = parsed.kind === 'edge';
    const depth = formContainerDepth(form, landingId);
    const isDeeper = depth > winnerDepth;
    const winsTie = depth === winnerDepth && isEdge && !winnerIsEdge;

    if (winner === null || isDeeper || winsTie) {
      winner = id;
      winnerDepth = depth;
      winnerIsEdge = isEdge;
    }
  }

  return winner ?? String(collisions[0]!.id);
};

/**
 * Resolves a drop target to a `moveFormElementTo` call, or null when the move
 * is disallowed (dropping onto self, into own subtree, or an unresolvable
 * target). The reducer re-validates and no-ops on any of these too — this is
 * the same guard duplicated at the droppable level so drop zones for invalid
 * targets don't render mid-drag.
 */
export const resolveFormDrop = (
  form: WorkflowForm,
  activeId: string,
  target: FormDropTarget
): { parentId: string; index: number } | null => {
  if (target.kind === 'into') {
    const container = form.elements[target.containerId];
    if (container?.type !== 'container' || isFormDescendantOrSelf(form, activeId, target.containerId)) {
      return null;
    }
    return { index: container.data.children.length, parentId: target.containerId };
  }

  if (target.elementId === activeId || isFormDescendantOrSelf(form, activeId, target.elementId)) {
    return null;
  }
  const parentId = findFormParentId(form, target.elementId);
  if (parentId === null) {
    return null;
  }
  const parent = form.elements[parentId];
  if (parent?.type !== 'container') {
    return null;
  }
  const index = parent.data.children.indexOf(target.elementId);
  if (index < 0) {
    return null;
  }
  return { index: target.edge === 'above' ? index : index + 1, parentId };
};
