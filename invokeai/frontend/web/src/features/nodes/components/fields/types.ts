import { InputField } from 'features/nodes/types';

export type FieldComponentProps<T extends InputField> = {
  nodeId: string;
  field: T;
};
