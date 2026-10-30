import type { WorkflowFormElement } from '@features/workflow/contracts';

type BodyElement = Exclude<WorkflowFormElement, { type: 'heading' }>;
type HeadingElement = Extract<WorkflowFormElement, { type: 'heading' }>;

export type FormSectionItem =
  | { kind: 'element'; element: BodyElement }
  | { kind: 'section'; heading: HeadingElement; body: BodyElement[] };

const SECTION_STATE_KEY = 'formSectionsOpen';

/** Headings group only siblings; dividers end sections and only redundant inter-section lines are omitted. */
export const groupFormSections = (elements: readonly WorkflowFormElement[]): FormSectionItem[] => {
  const groups: FormSectionItem[] = [];
  let current: Extract<FormSectionItem, { kind: 'section' }> | null = null;

  for (const [index, element] of elements.entries()) {
    if (element.type === 'heading') {
      if (current) {
        groups.push(current);
      }
      current = { kind: 'section', heading: element, body: [] };
    } else if (element.type === 'divider') {
      const separatesAdjacentSections = current !== null && elements[index + 1]?.type === 'heading';
      if (current) {
        groups.push(current);
        current = null;
      }
      // The gap replaces a divider used only between two collapsible headings; other authored dividers stay visible.
      if (!separatesAdjacentSections) {
        groups.push({ kind: 'element', element });
      }
    } else if (current) {
      current.body.push(element);
    } else {
      groups.push({ kind: 'element', element });
    }
  }

  if (current) {
    groups.push(current);
  }
  return groups;
};

/** Widget workspace values are personal and project-scoped, unlike the shared authored workflow form. */
export const getFormSectionOpenState = (
  workflowValues: Record<string, unknown> | undefined,
  workflowId: string
): Record<string, boolean> => {
  const all = workflowValues?.[SECTION_STATE_KEY];
  const current = isRecord(all) ? all[workflowId] : null;
  if (!isRecord(current)) {
    return {};
  }
  return Object.fromEntries(
    Object.entries(current).filter((entry): entry is [string, boolean] => typeof entry[1] === 'boolean')
  );
};

export const updateFormSectionOpenState = (
  workflowValues: Record<string, unknown> | undefined,
  workflowId: string,
  sectionId: string,
  open: boolean
): Record<string, unknown> => {
  const stored = workflowValues?.[SECTION_STATE_KEY];
  const all = isRecord(stored) ? stored : {};
  return {
    [SECTION_STATE_KEY]: {
      ...all,
      [workflowId]: { ...getFormSectionOpenState(workflowValues, workflowId), [sectionId]: open },
    },
  };
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
