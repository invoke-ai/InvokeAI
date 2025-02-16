import type { FieldInputInstance, FieldInputTemplate } from 'features/nodes/types/field';

export type FieldComponentProps<V extends FieldInputInstance, T extends FieldInputTemplate, C = void> = {
  nodeId: string;
  field: V;
  fieldTemplate: T;
} & Omit<C, 'nodeId' | 'field' | 'fieldTemplate'>;
