type SerializableValue = string | number | boolean | null | undefined | SerializableValue[] | SerializableObject;
export type SerializableObject = {
  [k: string | number]: SerializableValue;
};
