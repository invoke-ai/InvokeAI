import { apiFetchJson, getApiErrorMessage } from '@workbench/backend/http';
import { createExternalStore } from '@workbench/externalStore';

import type {
  FieldInputTemplate,
  FieldOutputTemplate,
  FieldType,
  InvocationTemplate,
  InvocationTemplates,
} from './types';

/**
 * Invocation templates parsed from the backend OpenAPI schema. They are
 * session-lived, backend-owned data shared by every workflow surface, so they
 * live in an external store (the same pattern as the models library) rather
 * than in project state.
 */

export interface InvocationTemplatesSnapshot {
  status: 'idle' | 'loading' | 'loaded' | 'error';
  error: string | null;
  templates: InvocationTemplates;
}

const store = createExternalStore<InvocationTemplatesSnapshot>({ error: null, status: 'idle', templates: {} });

type JsonObject = Record<string, unknown>;

const isJsonObject = (value: unknown): value is JsonObject => typeof value === 'object' && value !== null;

/** Invocations that exist in the schema but are not user-placeable nodes. */
const INVOCATION_DENYLIST = new Set(['graph', 'linear_ui_output']);

const RESERVED_INPUT_FIELD_NAMES = new Set(['id', 'type', 'use_cache', 'is_intermediate']);
const RESERVED_FIELD_TYPE_NAMES = new Set(['IsIntermediate']);

const OPENAPI_TO_FIELD_TYPE_MAP: Record<string, string> = {
  boolean: 'BooleanField',
  integer: 'IntegerField',
  number: 'FloatField',
  string: 'StringField',
};

const COLLECTION_OVERRIDE_TYPE_NAMES = new Set(['CollectionField']);

const refToSchemaName = (ref: unknown): string | null => {
  if (typeof ref !== 'string') {
    return null;
  }

  return ref.split('/').at(-1) ?? null;
};

const getRef = (schema: JsonObject): string | null => refToSchemaName(schema.$ref);

/**
 * Derives the field type from an OpenAPI property schema. Ported from the
 * legacy `parseFieldType`; returns null instead of throwing so unparseable
 * fields are skipped rather than failing the whole template.
 */
export const parseFieldType = (schema: unknown): FieldType | null => {
  if (!isJsonObject(schema)) {
    return null;
  }

  const ref = getRef(schema);

  if (ref) {
    return { batch: false, cardinality: 'SINGLE', name: ref };
  }

  // `Literal["value"]` pydantic fields arrive as `const` — treated as an enum.
  if (schema.const !== undefined || schema.enum !== undefined) {
    return { batch: false, cardinality: 'SINGLE', name: 'EnumField' };
  }

  if (schema.type === undefined) {
    if (Array.isArray(schema.allOf) && isJsonObject(schema.allOf[0])) {
      const name = getRef(schema.allOf[0]);

      return name ? { batch: false, cardinality: 'SINGLE', name } : null;
    }

    if (Array.isArray(schema.anyOf)) {
      const variants = schema.anyOf.filter(
        (variant): variant is JsonObject => isJsonObject(variant) && variant.type !== 'null'
      );

      if (variants.length === 1) {
        return parseFieldType(variants[0]);
      }

      // `T | list[T]` unions become SINGLE_OR_COLLECTION of the base type.
      if (variants.length === 2) {
        const arrayVariant = variants.find((variant) => variant.type === 'array');
        const itemVariant = variants.find((variant) => variant.type !== 'array');

        if (arrayVariant && itemVariant && isJsonObject(arrayVariant.items)) {
          const arrayItemName =
            getRef(arrayVariant.items) ??
            (typeof arrayVariant.items.type === 'string' ? arrayVariant.items.type : null);
          const itemName = getRef(itemVariant) ?? (typeof itemVariant.type === 'string' ? itemVariant.type : null);

          if (arrayItemName && arrayItemName === itemName) {
            return {
              batch: false,
              cardinality: 'SINGLE_OR_COLLECTION',
              name: OPENAPI_TO_FIELD_TYPE_MAP[itemName] ?? itemName,
            };
          }
        }
      }

      return null;
    }

    return null;
  }

  if (schema.type === 'array') {
    if (!isJsonObject(schema.items)) {
      return null;
    }

    const itemRef = getRef(schema.items);

    if (itemRef) {
      return { batch: false, cardinality: 'COLLECTION', name: itemRef };
    }

    const itemType = typeof schema.items.type === 'string' ? OPENAPI_TO_FIELD_TYPE_MAP[schema.items.type] : undefined;

    return itemType ? { batch: false, cardinality: 'COLLECTION', name: itemType } : null;
  }

  if (typeof schema.type === 'string') {
    const name = OPENAPI_TO_FIELD_TYPE_MAP[schema.type];

    return name ? { batch: false, cardinality: 'SINGLE', name } : null;
  }

  return null;
};

const getUiTypeOverride = (property: JsonObject): FieldType | null => {
  if (typeof property.ui_type !== 'string') {
    return null;
  }

  return {
    batch: false,
    cardinality: COLLECTION_OVERRIDE_TYPE_NAMES.has(property.ui_type) ? 'COLLECTION' : 'SINGLE',
    name: property.ui_type,
  };
};

const startCase = (value: string): string =>
  value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
    .trim();

const getNumberOrNull = (value: unknown): number | null => (typeof value === 'number' ? value : null);

const getStringArrayOrNull = (value: unknown): string[] | null =>
  Array.isArray(value) && value.every((item) => typeof item === 'string') ? value : null;

const getDefaultValueForType = (type: FieldType, options: string[] | null): unknown => {
  if (type.cardinality === 'COLLECTION') {
    return undefined;
  }

  switch (type.name) {
    case 'StringField':
      return '';
    case 'IntegerField':
    case 'FloatField':
      return 0;
    case 'BooleanField':
      return false;
    case 'EnumField':
      return options?.[0];
    default:
      return undefined;
  }
};

const buildInputTemplate = (name: string, property: JsonObject, type: FieldType): FieldInputTemplate => {
  const enumValues = Array.isArray(property.enum)
    ? property.enum
    : property.const !== undefined
      ? [property.const]
      : null;
  const options = enumValues ? enumValues.filter((value): value is string => typeof value === 'string') : null;
  const input = property.input === 'connection' || property.input === 'direct' ? property.input : 'any';
  const uiChoiceLabels = isJsonObject(property.ui_choice_labels)
    ? Object.fromEntries(
        Object.entries(property.ui_choice_labels).filter(
          (entry): entry is [string, string] => typeof entry[1] === 'string'
        )
      )
    : null;

  return {
    default: property.default ?? getDefaultValueForType(type, options),
    description: typeof property.description === 'string' ? property.description : '',
    exclusiveMaximum: getNumberOrNull(property.exclusiveMaximum),
    exclusiveMinimum: getNumberOrNull(property.exclusiveMinimum),
    input,
    maximum: getNumberOrNull(property.maximum),
    minimum: getNumberOrNull(property.minimum),
    multipleOf: getNumberOrNull(property.multipleOf),
    name,
    options,
    required: property.orig_required === true,
    title: typeof property.title === 'string' ? property.title : startCase(name),
    type,
    uiChoiceLabels,
    uiComponent:
      property.ui_component === 'slider' || property.ui_component === 'textarea' ? property.ui_component : null,
    uiHidden: property.ui_hidden === true,
    uiModelBase: getStringArrayOrNull(property.ui_model_base),
    uiModelType: getStringArrayOrNull(property.ui_model_type),
    uiOrder: getNumberOrNull(property.ui_order),
  };
};

const parseFieldProperty = (property: JsonObject): FieldType | null => {
  const override = getUiTypeOverride(property);
  const parsed = parseFieldType(property);

  if (override) {
    if (parsed && (parsed.name !== override.name || parsed.cardinality !== override.cardinality)) {
      override.originalType = parsed;
    }

    return override;
  }

  return parsed;
};

const parseInvocationSchema = (schema: JsonObject, schemas: JsonObject): InvocationTemplate | null => {
  const properties = isJsonObject(schema.properties) ? schema.properties : null;
  const typeProperty = properties && isJsonObject(properties.type) ? properties.type : null;
  const type = typeProperty && typeof typeProperty.default === 'string' ? typeProperty.default : null;

  if (!properties || !type || INVOCATION_DENYLIST.has(type)) {
    return null;
  }

  const inputs: Record<string, FieldInputTemplate> = {};

  for (const [name, rawProperty] of Object.entries(properties)) {
    if (
      RESERVED_INPUT_FIELD_NAMES.has(name) ||
      (type === 'iterate' && name === 'index') ||
      !isJsonObject(rawProperty)
    ) {
      continue;
    }

    if (rawProperty.field_kind !== 'input') {
      continue;
    }

    const fieldType = parseFieldProperty(rawProperty);

    if (!fieldType || RESERVED_FIELD_TYPE_NAMES.has(fieldType.name)) {
      continue;
    }

    inputs[name] = buildInputTemplate(name, rawProperty, fieldType);
  }

  const outputRefName = isJsonObject(schema.output) ? getRef(schema.output) : null;
  const outputSchema =
    outputRefName && isJsonObject(schemas[outputRefName]) ? (schemas[outputRefName] as JsonObject) : null;
  const outputProperties = outputSchema && isJsonObject(outputSchema.properties) ? outputSchema.properties : null;

  if (!outputProperties) {
    return null;
  }

  const outputTypeProperty = isJsonObject(outputProperties.type) ? outputProperties.type : null;
  const outputType =
    outputTypeProperty && typeof outputTypeProperty.default === 'string' ? outputTypeProperty.default : '';
  const outputs: Record<string, FieldOutputTemplate> = {};

  for (const [name, rawProperty] of Object.entries(outputProperties)) {
    if (name === 'type' || !isJsonObject(rawProperty) || rawProperty.field_kind !== 'output') {
      continue;
    }

    const fieldType = parseFieldProperty(rawProperty);

    if (!fieldType) {
      continue;
    }

    outputs[name] = {
      description: typeof rawProperty.description === 'string' ? rawProperty.description : '',
      name,
      title: typeof rawProperty.title === 'string' ? rawProperty.title : startCase(name),
      type: fieldType,
    };
  }

  const useCacheProperty = isJsonObject(properties.use_cache) ? properties.use_cache : null;

  return {
    category: typeof schema.category === 'string' ? schema.category : 'other',
    classification: typeof schema.classification === 'string' ? schema.classification : 'stable',
    description: typeof schema.description === 'string' ? schema.description : '',
    inputs,
    nodePack: typeof schema.node_pack === 'string' ? schema.node_pack : 'invokeai',
    outputs,
    outputType,
    tags: getStringArrayOrNull(schema.tags) ?? [],
    title: typeof schema.title === 'string' ? schema.title.replace('Invocation', '').trim() : type,
    type,
    useCache: useCacheProperty?.default !== false,
    version: typeof schema.version === 'string' ? schema.version : '1.0.0',
  };
};

/** Parses a full OpenAPI document into invocation templates. Exported for tests. */
export const parseOpenApiToTemplates = (openApiDocument: unknown): InvocationTemplates => {
  if (!isJsonObject(openApiDocument)) {
    return {};
  }

  const components = isJsonObject(openApiDocument.components) ? openApiDocument.components : null;
  const schemas = components && isJsonObject(components.schemas) ? components.schemas : null;

  if (!schemas) {
    return {};
  }

  const templates: InvocationTemplates = {};

  for (const schema of Object.values(schemas)) {
    if (!isJsonObject(schema) || schema.class !== 'invocation') {
      continue;
    }

    const template = parseInvocationSchema(schema, schemas);

    if (template) {
      templates[template.type] = template;
    }
  }

  return templates;
};

export const refreshInvocationTemplates = async (): Promise<void> => {
  store.patchSnapshot({ error: null, status: 'loading' });

  try {
    // FastAPI serves the schema at the app root (proxied in dev), not under /api.
    const openApiDocument = await apiFetchJson<unknown>('/openapi.json');

    store.setSnapshot({ error: null, status: 'loaded', templates: parseOpenApiToTemplates(openApiDocument) });
  } catch (error) {
    store.patchSnapshot({
      error: getApiErrorMessage(error, 'Failed to load node definitions from the backend.'),
      status: 'error',
    });
  }
};

export const ensureInvocationTemplatesLoaded = (): void => {
  const { status } = store.getSnapshot();

  if (status === 'idle' || status === 'error') {
    void refreshInvocationTemplates();
  }
};

export const useInvocationTemplatesSnapshot = (): InvocationTemplatesSnapshot => store.useSnapshot();

export const useInvocationTemplatesSelector = store.useSelector;

/** Imperative read for the workbench reducer and route validation. */
export const getInvocationTemplatesSnapshot = (): InvocationTemplatesSnapshot => store.getSnapshot();
