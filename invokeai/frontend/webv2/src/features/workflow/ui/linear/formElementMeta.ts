import type { WorkflowFormElement } from '@features/workflow/contracts';
import type { LucideIcon } from 'lucide-react';

import { Columns2Icon, CrosshairIcon, HeadingIcon, MinusIcon, Rows2Icon, TextIcon } from 'lucide-react';

/**
 * One name and one icon per form element kind.
 *
 * The builder shows an element's identity in three places — the card title bar,
 * the drag ghost, and the Add menu — and they were each spelling it out
 * separately, so the menu could offer "Container (column)" while the card it
 * produced called itself something else. Naming a kind is one fact; it belongs
 * in one table.
 *
 * Containers are keyed by layout rather than by type alone: a row and a column
 * are the two things a person is actually choosing between, so they carry
 * distinct names and distinct icons.
 */
export type FormElementMetaKey = 'container-column' | 'container-row' | 'divider' | 'heading' | 'node-field' | 'text';

export interface FormElementMeta {
  icon: LucideIcon;
  label: string;
}

export const FORM_ELEMENT_META: Record<FormElementMetaKey, FormElementMeta> = {
  'container-column': { icon: Columns2Icon, label: 'Container (column)' },
  'container-row': { icon: Rows2Icon, label: 'Container (row)' },
  divider: { icon: MinusIcon, label: 'Divider' },
  heading: { icon: HeadingIcon, label: 'Heading' },
  'node-field': { icon: CrosshairIcon, label: 'Node Field' },
  text: { icon: TextIcon, label: 'Text' },
};

/**
 * The kinds the Add menu offers, in the order it offers them.
 *
 * `node-field` is absent by construction: a field enters a form by being
 * dragged off the node that owns it, so there is nothing for a menu entry to
 * create.
 */
export const ADDABLE_FORM_ELEMENT_KEYS = [
  'heading',
  'text',
  'divider',
  'container-column',
  'container-row',
] as const satisfies readonly FormElementMetaKey[];

export type AddableFormElementKey = (typeof ADDABLE_FORM_ELEMENT_KEYS)[number];

export const getFormElementMetaKey = (element: WorkflowFormElement): FormElementMetaKey =>
  element.type === 'container' ? (element.data.layout === 'row' ? 'container-row' : 'container-column') : element.type;

/** Title shown in a card's title bar and the drag ghost. Shared so the two never drift. */
export const getFormElementTitle = (element: WorkflowFormElement): string =>
  FORM_ELEMENT_META[getFormElementMetaKey(element)].label;
