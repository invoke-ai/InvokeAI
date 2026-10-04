import type {
  FieldInputTemplate,
  FieldOutputTemplate,
  FieldType,
  InvocationTemplate,
  InvocationTemplates,
  InvocationTemplatesSnapshot,
  ProjectGraphState,
} from '@features/workflow/core/types';

import {
  getDefaultWorkflowGeneratorValue,
  getWorkflowBatchCollectionField,
  getWorkflowGeneratorOutputField,
} from '@features/workflow/core/batch';
import { updateWorkflowNodes } from '@features/workflow/core/document';
import { isEditableCollectionFieldType } from '@features/workflow/core/fields';
import { createLogger } from '@platform/logging/logger';
import {
  captureAccountScope,
  isAccountScopeCurrent,
  registerAccountOwnedResource,
} from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { apiFetchJson, getApiErrorMessage } from '@platform/transport/http';

export type { InvocationTemplatesSnapshot } from '@features/workflow/core/types';

/** Share backend invocation templates in session-lived external state rather than project documents. */

const EMPTY_INVOCATION_TEMPLATES: InvocationTemplatesSnapshot = { error: null, status: 'idle', templates: {} };
const store = createExternalStore<InvocationTemplatesSnapshot>(EMPTY_INVOCATION_TEMPLATES);

registerAccountOwnedResource({
  clear: () => {
    store.setSnapshot(EMPTY_INVOCATION_TEMPLATES);
  },
  name: 'invocation-templates',
});

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

/** Return null for unparseable fields so one unsupported property cannot invalidate the whole template. */
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

const getEnumValues = (property: JsonObject): unknown[] | null => {
  if (property.enum !== undefined) {
    return Array.isArray(property.enum) ? property.enum : [];
  }

  if (property.const !== undefined) {
    return [property.const];
  }

  if (Array.isArray(property.anyOf)) {
    const variants = property.anyOf.filter(
      (variant): variant is JsonObject => isJsonObject(variant) && variant.type !== 'null'
    );

    if (variants.length === 1) {
      return getEnumValues(variants[0]);
    }
  }

  return null;
};

const getDefaultValueForType = (type: FieldType, options: unknown[] | null): unknown => {
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

/** The array schema of a list property: the property itself, or the array branch of an `Optional[list[...]]`. */
const getArraySchema = (property: JsonObject): JsonObject | null => {
  if (property.type === 'array') {
    return property;
  }

  if (Array.isArray(property.anyOf)) {
    const arrays = property.anyOf.filter(
      (variant): variant is JsonObject => isJsonObject(variant) && variant.type === 'array'
    );

    return arrays.length === 1 ? (arrays[0] as JsonObject) : null;
  }

  return null;
};

const buildInputTemplate = (
  name: string,
  property: JsonObject,
  type: FieldType,
  fieldKind: FieldInputTemplate['fieldKind']
): FieldInputTemplate => {
  const enumValues = getEnumValues(property);
  const arraySchema = type.cardinality === 'COLLECTION' ? getArraySchema(property) : null;
  // pydantic places a list's item constraints on `items`; scalar constraints stay on the property.
  const constraints = arraySchema && isJsonObject(arraySchema.items) ? arraySchema.items : property;
  const options = enumValues
    ? enumValues.every(
        (value) =>
          typeof value === 'string' ||
          (typeof value === 'number' && Number.isFinite(value)) ||
          typeof value === 'boolean'
      )
      ? enumValues
      : []
    : null;
  const input = property.input === 'connection' || property.input === 'direct' ? property.input : 'any';
  const uiChoiceLabels = isJsonObject(property.ui_choice_labels)
    ? Object.fromEntries(
        Object.entries(property.ui_choice_labels).filter(
          (entry): entry is [string, string] => typeof entry[1] === 'string'
        )
      )
    : null;

  const required = property.orig_required === true;

  return {
    default:
      type.name === 'EnumField' && property.default === null && !required
        ? undefined
        : property.default !== undefined && property.default !== null
          ? property.default
          : // A required editable list starts empty so the widget has something to append to; a list
            // without a widget stays absent so readiness still asks for its connection.
            required && input !== 'connection' && isEditableCollectionFieldType(type)
            ? []
            : // The backend's generator models are empty; the editor owns their shape and their default.
              (getDefaultWorkflowGeneratorValue(type.name) ?? getDefaultValueForType(type, options)),
    description: typeof property.description === 'string' ? property.description : '',
    exclusiveMaximum: getNumberOrNull(constraints.exclusiveMaximum),
    exclusiveMinimum: getNumberOrNull(constraints.exclusiveMinimum),
    fieldKind,
    input,
    maximum: getNumberOrNull(constraints.maximum),
    minimum: getNumberOrNull(constraints.minimum),
    multipleOf: getNumberOrNull(constraints.multipleOf),
    name,
    options,
    required,
    ...(arraySchema
      ? {
          maxItems: getNumberOrNull(arraySchema.maxItems),
          maxLength: getNumberOrNull(constraints.maxLength),
          minItems: getNumberOrNull(arraySchema.minItems),
          minLength: getNumberOrNull(constraints.minLength),
        }
      : {}),
    title: typeof property.title === 'string' ? property.title : startCase(name),
    type,
    uiChoiceLabels,
    uiComponent:
      property.ui_component === 'slider' ||
      property.ui_component === 'textarea' ||
      property.ui_component === 'video-frame-index'
        ? property.ui_component
        : null,
    uiHidden: property.ui_hidden === true,
    uiModelBase: getStringArrayOrNull(property.ui_model_base),
    uiModelFormat: getStringArrayOrNull(property.ui_model_format),
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

    // Retain internal metadata/board inputs for edges and Linear controls. Exclude node attributes; queue routing
    // still determines final result boards.
    const isInternal = rawProperty.field_kind === 'internal';

    if (rawProperty.field_kind !== 'input' && !isInternal) {
      continue;
    }

    const fieldType = parseFieldProperty(rawProperty);

    if (!fieldType || RESERVED_FIELD_TYPE_NAMES.has(fieldType.name)) {
      continue;
    }

    // A batch node's list only accepts a generator, and a generator's list only feeds a batch node.
    if (getWorkflowBatchCollectionField(type) === name) {
      fieldType.batch = true;
    }

    inputs[name] = buildInputTemplate(name, rawProperty, fieldType, isInternal ? 'internal' : 'input');
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

    if (getWorkflowGeneratorOutputField(type) === name) {
      fieldType.batch = true;
    }

    outputs[name] = {
      description: typeof rawProperty.description === 'string' ? rawProperty.description : '',
      name,
      outputScope:
        rawProperty.output_scope === 'iteration' || rawProperty.output_scope === 'final'
          ? rawProperty.output_scope
          : undefined,
      title: typeof rawProperty.title === 'string' ? rawProperty.title : startCase(name),
      type: fieldType,
      uiHidden: rawProperty.ui_hidden === true,
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
  const owner = captureAccountScope();

  store.patchSnapshot({ error: null, status: 'loading' });

  try {
    // FastAPI serves the schema at the app root (proxied in dev), not under /api.
    const openApiDocument = await apiFetchJson<unknown>('/openapi.json', { signal: owner.signal });

    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    store.setSnapshot({ error: null, status: 'loaded', templates: parseOpenApiToTemplates(openApiDocument) });
  } catch (error) {
    if (!isAccountScopeCurrent(owner)) {
      return;
    }

    createLogger({ area: 'templates', namespace: 'workflows' }).error({
      error,
      message: 'Failed to load node definitions',
      name: 'workflows.templates-load-failed',
    });
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

/**
 * Moves a freshly parsed document's nodes to the loaded templates before it enters the project, so an outdated
 * workflow opens current, and words what the update could not keep. A document loaded before templates arrive is
 * left as is; the editor's update actions cover it later.
 */
export const updateLoadedWorkflowNodes = (
  document: ProjectGraphState,
  translate: (key: string, options: { count: number }) => string
): { document: ProjectGraphState; warnings: string[] } => {
  const snapshot = store.getSnapshot();

  if (snapshot.status !== 'loaded') {
    return { document, warnings: [] };
  }

  const update = updateWorkflowNodes(document, snapshot.templates);
  const warnings = [
    ...(update.skippedNodeIds.length > 0
      ? [translate('nodes.unableToUpdateNodes', { count: update.skippedNodeIds.length })]
      : []),
    ...(update.droppedEdgeIds.length > 0
      ? [translate('nodes.updateDroppedEdges', { count: update.droppedEdgeIds.length })]
      : []),
    ...(update.droppedFormElementIds.length > 0
      ? [translate('nodes.updateDroppedFormFields', { count: update.droppedFormElementIds.length })]
      : []),
  ];

  return { document: update.document, warnings };
};

/** For readers that combine this store with another one; a single-store reader uses the selector. */
export const subscribeInvocationTemplates = store.subscribe;
