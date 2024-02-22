export type JSONValue = string | number | boolean | null | JSONValue[] | { [key: string]: JSONValue };

export interface JSONObject {
  [k: string]: JSONValue;
}

export interface JSONArray extends Array<JSONValue> {}
