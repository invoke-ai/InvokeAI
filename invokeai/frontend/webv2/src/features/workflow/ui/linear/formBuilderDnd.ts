import type { WorkflowForm } from '@features/workflow/contracts';

/**
 * Resolve drop IDs into pure reparent commands; use maintained parentId links rather than scanning containers and
 * commit at drag end.
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

/** Compare landing containers: into:X targets X, while edge:X targets X's parent. */
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
 * Choose deepest landing container, then prefer precise edge insertion at equal depth. Independent depth
 * calculation makes collision ordering irrelevant.
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
 * Reject self, descendant, and unresolved drops before rendering affordances; reducer validation independently
 * enforces the same move constraints.
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
