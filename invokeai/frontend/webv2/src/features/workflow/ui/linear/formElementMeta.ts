import type { WorkflowFormElement } from '@features/workflow/contracts';
import type { LucideIcon } from 'lucide-react';

import { Columns2Icon, CrosshairIcon, HeadingIcon, MinusIcon, Rows2Icon, TextIcon } from 'lucide-react';

/** Share names/icons across cards, ghosts, and Add menus; row and column containers have distinct identities. */
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

/** Exclude node-field from Add because fields enter forms by dragging from their owning nodes. */
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
