import type { WorkflowFormElement } from '@features/workflow/contracts';
import type { LucideIcon } from 'lucide-react';

import { Columns2Icon, CrosshairIcon, HeadingIcon, MinusIcon, Rows2Icon, TextIcon } from 'lucide-react';

/** Share names/icons across cards, ghosts, and Add menus; row and column containers have distinct identities. */
export type FormElementMetaKey = 'container-column' | 'container-row' | 'divider' | 'heading' | 'node-field' | 'text';

export interface FormElementMeta {
  icon: LucideIcon;
  /** Translation key of the element's display name. */
  labelKey: string;
}

export const FORM_ELEMENT_META: Record<FormElementMetaKey, FormElementMeta> = {
  'container-column': { icon: Columns2Icon, labelKey: 'widgets.workflow.formBuilder.elements.containerColumn' },
  'container-row': { icon: Rows2Icon, labelKey: 'widgets.workflow.formBuilder.elements.containerRow' },
  divider: { icon: MinusIcon, labelKey: 'widgets.workflow.formBuilder.elements.divider' },
  heading: { icon: HeadingIcon, labelKey: 'widgets.workflow.formBuilder.elements.heading' },
  'node-field': { icon: CrosshairIcon, labelKey: 'widgets.workflow.formBuilder.elements.nodeField' },
  text: { icon: TextIcon, labelKey: 'widgets.workflow.formBuilder.elements.text' },
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

/** Translation key of the title shown in a card's title bar and the drag ghost. Shared so the two never drift. */
export const getFormElementTitleKey = (element: WorkflowFormElement): string =>
  FORM_ELEMENT_META[getFormElementMetaKey(element)].labelKey;
