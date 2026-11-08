import type { FieldInputInstance, FieldInputTemplate } from 'features/nodes/types/field';

export type FieldComponentProps<
  TFieldInstance extends FieldInputInstance,
  TFieldTemplate extends FieldInputTemplate,
  FieldSettings = void,
> = {
  nodeId: string;
  field: TFieldInstance;
  fieldTemplate: TFieldTemplate;
} & Omit<FieldSettings, 'nodeId' | 'field' | 'fieldTemplate'>;
