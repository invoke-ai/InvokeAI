import { z } from 'zod';

/**
 * Map of non-server-validated schemas to their server-validated counterparts.
 * Add entries here for any schemas that need to be replaced.
 */
const schemaReplacementMap = new Map<z.ZodType, z.ZodType>();

/**
 * Register a schema replacement mapping.
 * @param originalSchema The non-server-validated schema
 * @param serverValidatedSchema The server-validated replacement schema
 */
export function registerSchemaReplacement<T extends z.ZodType>(originalSchema: T, serverValidatedSchema: T): void {
  schemaReplacementMap.set(originalSchema, serverValidatedSchema);
}

export function clearSchemaReplacements(): void {
  schemaReplacementMap.clear();
}

/**
 * Recursively replaces non-server-validated schemas with server-validated ones.
 * Handles objects, arrays, unions, intersections, and other composite types.
 *
 * @param schema The schema to transform
 * @returns A new schema with server-validated replacements
 */
export function replaceWithServerValidatedSchemas<T extends z.ZodType>(schema: T): T {
  // Check if this schema has a direct replacement
  const replacement = schemaReplacementMap.get(schema);
  if (replacement) {
    return replacement as T;
  }

  // Access the internal definition
  const def = schema._zod.def;
  const type = def.type;

  // Handle different schema types
  if (type === 'object') {
    // For objects, recursively transform the shape
    const shape = (def as any).shape;
    if (!shape) {
      return schema;
    }

    const newShape: Record<string, z.ZodType> = {};
    for (const key in shape) {
      newShape[key] = replaceWithServerValidatedSchemas(shape[key]);
    }

    // Create a new object with the transformed shape
    const newSchema = z.object(newShape);

    // Preserve the original object configuration (strict/strip/passthrough)
    const config = (def as any).config;
    if (config?.type === 'strict') {
      return newSchema.strict();
    } else if (config?.type === 'loose') {
      return newSchema.passthrough();
    }

    return newSchema;
  }

  if (type === 'array') {
    // For arrays, transform the element type
    const element = (def as any).element;
    if (!element) {
      return schema;
    }

    const newElement = replaceWithServerValidatedSchemas(element);
    return z.array(newElement);
  }

  if (type === 'union') {
    // For unions, transform all options
    const options = (def as any).options;
    if (!options || !Array.isArray(options)) {
      return schema;
    }

    const newOptions = options.map((opt) => replaceWithServerValidatedSchemas(opt));
    return z.union(newOptions as [z.ZodType, z.ZodType, ...z.ZodType[]]);
  }

  if (type === 'intersection') {
    // For intersections, transform both sides
    const left = (def as any).left;
    const right = (def as any).right;
    if (!left || !right) {
      return schema;
    }

    const newLeft = replaceWithServerValidatedSchemas(left);
    const newRight = replaceWithServerValidatedSchemas(right);
    return z.intersection(newLeft, newRight);
  }

  if (type === 'optional') {
    // For optional, transform the inner type
    const inner = (def as any).inner;
    if (!inner) {
      return schema;
    }

    const newInner = replaceWithServerValidatedSchemas(inner);
    return newInner.optional();
  }

  if (type === 'nullable') {
    // For nullable, transform the inner type
    const inner = (def as any).inner;
    if (!inner) {
      return schema;
    }

    const newInner = replaceWithServerValidatedSchemas(inner);
    return newInner.nullable();
  }

  if (type === 'default') {
    // For default, transform the inner type and preserve default value
    const inner = (def as any).inner;
    const defaultValue = (def as any).defaultValue;
    if (!inner) {
      return schema;
    }

    const newInner = replaceWithServerValidatedSchemas(inner);
    return newInner.default(defaultValue);
  }

  if (type === 'catch') {
    // For catch, transform the inner type and preserve catch value
    const inner = (def as any).inner;
    const catchValue = (def as any).catchValue;
    if (!inner) {
      return schema;
    }

    const newInner = replaceWithServerValidatedSchemas(inner);
    return newInner.catch(catchValue);
  }

  if (type === 'readonly') {
    // For readonly, transform the inner type
    const inner = (def as any).inner;
    if (!inner) {
      return schema;
    }

    const newInner = replaceWithServerValidatedSchemas(inner);
    return newInner.readonly();
  }

  if (type === 'promise') {
    // For promise, transform the inner type
    const inner = (def as any).inner;
    if (!inner) {
      return schema;
    }

    const newInner = replaceWithServerValidatedSchemas(inner);
    return z.promise(newInner);
  }

  if (type === 'lazy') {
    // For lazy schemas, we need to wrap the getter function
    const getter = (def as any).getter;
    if (!getter) {
      return schema;
    }

    return z.lazy(() => replaceWithServerValidatedSchemas(getter()));
  }

  if (type === 'record') {
    // For records, transform the value type
    const valueType = (def as any).valueType;
    const keyType = (def as any).keyType;
    if (!valueType) {
      return schema;
    }

    const newValueType = replaceWithServerValidatedSchemas(valueType);

    if (keyType) {
      return z.record(keyType, newValueType);
    }
    return z.record(newValueType);
  }

  if (type === 'map') {
    // For maps, transform key and value types
    const keyType = (def as any).keyType;
    const valueType = (def as any).valueType;
    if (!keyType || !valueType) {
      return schema;
    }

    const newKeyType = replaceWithServerValidatedSchemas(keyType);
    const newValueType = replaceWithServerValidatedSchemas(valueType);
    return z.map(newKeyType, newValueType);
  }

  if (type === 'set') {
    // For sets, transform the value type
    const valueType = (def as any).valueType;
    if (!valueType) {
      return schema;
    }

    const newValueType = replaceWithServerValidatedSchemas(valueType);
    return z.set(newValueType);
  }

  if (type === 'tuple') {
    // For tuples, transform each item
    const items = (def as any).items;
    if (!items || !Array.isArray(items)) {
      return schema;
    }

    const newItems = items.map((item) => replaceWithServerValidatedSchemas(item));
    return z.tuple(newItems as [z.ZodType, ...z.ZodType[]]);
  }

  if (type === 'transform' || type === 'pipe') {
    // For transforms and pipes, we need to handle carefully
    // In v4, these might have different internal structure
    // For now, return as-is since transforming these could break functionality
    return schema;
  }

  // For primitive types and any unhandled types, return as-is
  return schema;
}
