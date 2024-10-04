import type { FieldInputInstance, FieldInputTemplate } from 'features/nodes/types/field';

export type FieldComponentProps<V extends FieldInputInstance, T extends FieldInputTemplate> = {
  nodeId: string;
  field: V;
  fieldTemplate: T;
};
