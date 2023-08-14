import {
  InputFieldTemplate,
  InputFieldValue,
  InvocationNodeData,
  InvocationTemplate,
} from 'features/nodes/types/types';

export type FieldComponentProps<
  V extends InputFieldValue,
  T extends InputFieldTemplate
> = {
  nodeData: InvocationNodeData;
  nodeTemplate: InvocationTemplate;
  field: V;
  fieldTemplate: T;
};
