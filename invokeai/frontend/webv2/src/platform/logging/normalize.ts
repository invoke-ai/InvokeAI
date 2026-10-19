import type { JsonValue, SerializedError } from './contracts';

/** Bounds keep one entry small enough that a burst cannot retain unbounded memory or block the caller. */
export const MAX_DEPTH = 6;
export const MAX_COLLECTION_ITEMS = 50;
export const MAX_STRING_LENGTH = 1_000;
export const MAX_STACK_LENGTH = 4_000;
export const MAX_ERROR_CAUSE_DEPTH = 3;
export const MAX_ENTRY_BYTES = 8 * 1024;

const SENSITIVE_KEY_PATTERN =
  /(?:^|-)(?:token|secret|password|passwd|authorization|cookie|credential|api-?key|bearer)s?(?:$|-)/;
const ABSOLUTE_URL_PATTERN = /^[a-z][a-z0-9+.-]*:\/\//i;

export interface NormalizedValue {
  truncated: boolean;
  value: JsonValue;
}

interface Walk {
  seen: WeakSet<object>;
  truncated: boolean;
}

/** Match whole words in camelCase, snake_case and kebab-case keys such as `refreshToken` or `api_key`. */
const isSensitiveKey = (key: string): boolean =>
  SENSITIVE_KEY_PATTERN.test(
    key
      .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
      .replace(/[_.\s]+/g, '-')
      .toLowerCase()
  );

/** Keep an origin and path of a bare URL; drop query strings and fragments, which carry tokens and user input. */
export const stripUrlSecrets = (value: string): string => {
  if (/\s/.test(value)) {
    return value;
  }

  if (ABSOLUTE_URL_PATTERN.test(value)) {
    try {
      const url = new URL(value);

      return `${url.origin}${url.pathname}`;
    } catch {
      return value;
    }
  }

  if (value.startsWith('/')) {
    const cut = value.search(/[?#]/);

    return cut === -1 ? value : value.slice(0, cut);
  }

  return value;
};

const normalizeString = (value: string, walk: Walk): string => {
  const stripped = stripUrlSecrets(value);

  if (stripped.length > MAX_STRING_LENGTH) {
    walk.truncated = true;

    return `${stripped.slice(0, MAX_STRING_LENGTH)}…[+${stripped.length - MAX_STRING_LENGTH} chars]`;
  }

  return stripped;
};

const describeBinary = (value: object): string | null => {
  if (typeof ArrayBuffer !== 'undefined' && (value instanceof ArrayBuffer || ArrayBuffer.isView(value))) {
    return `[binary ${value.byteLength} bytes]`;
  }

  if (typeof Blob !== 'undefined' && value instanceof Blob) {
    return `[blob ${value.size} bytes${value.type ? ` ${value.type}` : ''}]`;
  }

  if (typeof ImageData !== 'undefined' && value instanceof ImageData) {
    return `[image-data ${value.width}x${value.height}]`;
  }

  return null;
};

const describeDom = (value: object): string | null => {
  if (typeof Node !== 'undefined' && value instanceof Node) {
    return `[dom ${value.nodeName.toLowerCase()}]`;
  }

  if (typeof Window !== 'undefined' && value instanceof Window) {
    return '[window]';
  }

  if (typeof Event !== 'undefined' && value instanceof Event) {
    return `[event ${value.type}]`;
  }

  return null;
};

const getPrototypeName = (value: object): string => {
  const prototype = Object.getPrototypeOf(value) as { constructor?: { name?: string } } | null;

  return prototype?.constructor?.name ?? 'Object';
};

const readOwnDataProperty = (value: object, key: string): { ok: true; value: unknown } | { ok: false } => {
  try {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);

    if (!descriptor || !('value' in descriptor)) {
      return { ok: false };
    }

    return { ok: true, value: descriptor.value };
  } catch {
    return { ok: false };
  }
};

const normalizeValue = (value: unknown, depth: number, walk: Walk): JsonValue => {
  switch (typeof value) {
    case 'string':
      return normalizeString(value, walk);
    case 'number':
      return Number.isFinite(value) ? value : String(value);
    case 'boolean':
      return value;
    case 'bigint':
      return `${value}n`;
    case 'undefined':
      return null;
    case 'symbol':
      return value.toString();
    case 'function':
      return `[function ${value.name || 'anonymous'}]`;
    case 'object':
      break;
  }

  if (value === null) {
    return null;
  }

  const object = value as object;

  if (object instanceof Error) {
    if (walk.seen.has(object)) {
      return '[circular]';
    }

    walk.seen.add(object);
    try {
      return serializeErrorAtDepth(object, depth, walk) as unknown as JsonValue;
    } finally {
      walk.seen.delete(object);
    }
  }

  if (object instanceof Date) {
    return Number.isNaN(object.getTime()) ? '[invalid date]' : object.toISOString();
  }

  if (object instanceof RegExp) {
    return object.toString();
  }

  const binary = describeBinary(object) ?? describeDom(object);

  if (binary) {
    return binary;
  }

  if (walk.seen.has(object)) {
    return '[circular]';
  }

  if (depth >= MAX_DEPTH) {
    walk.truncated = true;

    return '[depth limit]';
  }

  walk.seen.add(object);

  try {
    if (Array.isArray(object)) {
      return normalizeArray(object, depth, walk);
    }

    if (object instanceof Map) {
      return normalizeArray(
        [...object.entries()].map(([key, entry]) => [
          key,
          typeof key === 'string' && isSensitiveKey(key) ? '[redacted]' : entry,
        ]),
        depth,
        walk
      );
    }

    if (object instanceof Set) {
      return normalizeArray([...object.values()], depth, walk);
    }

    return normalizeRecord(object, depth, walk);
  } finally {
    walk.seen.delete(object);
  }
};

const normalizeArray = (items: unknown[], depth: number, walk: Walk): JsonValue => {
  const output: JsonValue[] = [];

  for (const item of items.slice(0, MAX_COLLECTION_ITEMS)) {
    output.push(normalizeValue(item, depth + 1, walk));
  }

  if (items.length > MAX_COLLECTION_ITEMS) {
    walk.truncated = true;
    output.push(`[+${items.length - MAX_COLLECTION_ITEMS} more]`);
  }

  return output;
};

const normalizeRecord = (object: object, depth: number, walk: Walk): JsonValue => {
  const output: { [key: string]: JsonValue } = {};
  let keys: string[];

  try {
    keys = Object.keys(object);
  } catch {
    return `[unreadable ${getPrototypeName(object)}]`;
  }

  for (const key of keys.slice(0, MAX_COLLECTION_ITEMS)) {
    if (isSensitiveKey(key)) {
      output[key] = '[redacted]';
      continue;
    }

    const read = readOwnDataProperty(object, key);

    if (!read.ok) {
      output[key] = '[accessor]';
      continue;
    }

    if (read.value === undefined) {
      continue;
    }

    output[key] = normalizeValue(read.value, depth + 1, walk);
  }

  if (keys.length > MAX_COLLECTION_ITEMS) {
    walk.truncated = true;
    output['[truncated]'] = `+${keys.length - MAX_COLLECTION_ITEMS} keys`;
  }

  return output;
};

/** Convert any value into an immutable JSON-safe snapshot. Never throws. */
export const normalizeContext = (value: unknown): NormalizedValue => {
  const walk: Walk = { seen: new WeakSet(), truncated: false };

  try {
    const normalized = normalizeValue(value, 0, walk);

    return { truncated: walk.truncated, value: normalized };
  } catch (error) {
    return { truncated: true, value: `[unserializable: ${describeError(error)}]` };
  }
};

const readErrorField = (error: object, key: string): unknown => {
  const own = readOwnDataProperty(error, key);

  if (own.ok) {
    return own.value;
  }

  // Standard fields (`name`, `message`, `stack`) are inherited data properties or engine-installed accessors
  // on Error instances; reading them is safe.
  try {
    return (error as Record<string, unknown>)[key];
  } catch {
    return undefined;
  }
};

const serializeErrorAtDepth = (error: unknown, depth: number, walk: Walk): SerializedError | JsonValue => {
  if (!(error instanceof Error)) {
    return normalizeValue(error, depth, walk);
  }

  if (depth >= MAX_DEPTH) {
    walk.truncated = true;

    return '[depth limit]';
  }

  const name = readErrorField(error, 'name');
  const message = readErrorField(error, 'message');
  const stack = readErrorField(error, 'stack');
  // Domain fields are read as own data properties only, so subclass accessors never run.
  const statusRead = readOwnDataProperty(error, 'status');
  const codeRead = readOwnDataProperty(error, 'code');
  const status = statusRead.ok ? statusRead.value : undefined;
  const code = codeRead.ok ? codeRead.value : undefined;
  const serialized: SerializedError = {
    message: typeof message === 'string' ? normalizeString(message, walk) : '',
    name: typeof name === 'string' && name ? name : 'Error',
  };

  if (typeof stack === 'string' && stack) {
    if (stack.length > MAX_STACK_LENGTH) {
      walk.truncated = true;
      serialized.stack = `${stack.slice(0, MAX_STACK_LENGTH)}…`;
    } else {
      serialized.stack = stack;
    }
  }

  if (typeof status === 'number') {
    serialized.status = status;
  }

  if (typeof code === 'string' || typeof code === 'number') {
    serialized.code = code;
  }

  const causeRead = readOwnDataProperty(error, 'cause');

  if (causeRead.ok && causeRead.value !== undefined) {
    const cause = causeRead.value;

    if (depth >= MAX_ERROR_CAUSE_DEPTH) {
      walk.truncated = true;
      serialized.cause = '[cause depth limit]';
    } else if (typeof cause === 'object' && cause !== null && walk.seen.has(cause)) {
      serialized.cause = '[circular]';
    } else {
      walk.seen.add(error);
      try {
        serialized.cause = serializeErrorAtDepth(cause, depth + 1, walk);
      } finally {
        walk.seen.delete(error);
      }
    }
  }

  return serialized;
};

export interface NormalizedError {
  error: SerializedError;
  truncated: boolean;
}

/** Snapshot an error's name, message, stack, cause and status/code without retaining the instance. */
export const normalizeError = (error: unknown): NormalizedError => {
  const walk: Walk = { seen: new WeakSet(), truncated: false };

  try {
    const serialized = serializeErrorAtDepth(error, 0, walk);

    if (serialized && typeof serialized === 'object' && !Array.isArray(serialized) && 'message' in serialized) {
      return { error: serialized as SerializedError, truncated: walk.truncated };
    }

    return {
      error: { message: typeof serialized === 'string' ? serialized : JSON.stringify(serialized), name: 'NonError' },
      truncated: walk.truncated,
    };
  } catch {
    return { error: { message: '[unserializable error]', name: 'Error' }, truncated: true };
  }
};

export const serializeError = (error: unknown): SerializedError => normalizeError(error).error;

/** A readable one-line message for notifications; never `[object Object]`. */
export const describeError = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message || error.name;
  }

  if (typeof error === 'string') {
    return error;
  }

  if (error === undefined || error === null) {
    return '';
  }

  if (typeof error === 'object') {
    const message = readOwnDataProperty(error, 'message');

    if (message.ok && typeof message.value === 'string') {
      return message.value;
    }

    try {
      return JSON.stringify(normalizeContext(error).value);
    } catch {
      return '[unserializable error]';
    }
  }

  return String(error);
};

const utf8 = typeof TextEncoder === 'undefined' ? null : new TextEncoder();

/** UTF-8 size of the serialized value, so the ceiling holds for non-ASCII text. */
export const measureJsonBytes = (value: unknown): number => {
  try {
    const json = JSON.stringify(value) ?? '';

    return utf8 ? utf8.encode(json).length : json.length;
  } catch {
    return Number.POSITIVE_INFINITY;
  }
};

/** Bound a message like any other string: length cap and URL secret stripping. */
export const normalizeMessage = (message: string): { truncated: boolean; value: string } => {
  const walk: Walk = { seen: new WeakSet(), truncated: false };
  const value = normalizeString(message, walk);

  return { truncated: walk.truncated, value };
};
