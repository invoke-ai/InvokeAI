type JSONValue = string | number | boolean | null | JSONValue[] | { [key: string]: JSONValue };

export interface JSONObject {
  [k: string]: JSONValue;
}

type SerializableValue = string | number | boolean | null | undefined | SerializableValue[] | SerializableObject;
export type SerializableObject = {
  [k: string | number]: SerializableValue;
};
