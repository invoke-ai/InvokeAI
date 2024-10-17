type SerializableValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | SerializableValue[]
  | readonly SerializableValue[]
  | SerializableObject;
export type SerializableObject = {
  [k: string | number]: SerializableValue;
};
