import {
  InputFieldTemplate,
  InputFieldValue,
} from 'features/nodes/types/types';

export type FieldComponentProps<
  V extends InputFieldValue,
  T extends InputFieldTemplate
> = {
  nodeId: string;
  field: V;
  template: T;
};
