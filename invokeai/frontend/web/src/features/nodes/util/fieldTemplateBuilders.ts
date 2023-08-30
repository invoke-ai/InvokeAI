import { OpenAPIV3 } from 'openapi-types';
import {
  BooleanInputFieldTemplate,
  ClipInputFieldTemplate,
  CollectionInputFieldTemplate,
  CollectionItemInputFieldTemplate,
  ColorInputFieldTemplate,
  ConditioningInputFieldTemplate,
  ControlInputFieldTemplate,
  ControlNetModelInputFieldTemplate,
  DenoiseMaskInputFieldTemplate,
  EnumInputFieldTemplate,
  FieldType,
  FloatInputFieldTemplate,
  ImageCollectionInputFieldTemplate,
  ImageInputFieldTemplate,
  InputFieldTemplateBase,
  IntegerInputFieldTemplate,
  InvocationFieldSchema,
  InvocationSchemaObject,
  LatentsInputFieldTemplate,
  LoRAModelInputFieldTemplate,
  MainModelInputFieldTemplate,
  SDXLMainModelInputFieldTemplate,
  SDXLRefinerModelInputFieldTemplate,
  SchedulerInputFieldTemplate,
  StringInputFieldTemplate,
  UNetInputFieldTemplate,
  VaeInputFieldTemplate,
  VaeModelInputFieldTemplate,
} from '../types/types';

export type BaseFieldProperties = 'name' | 'title' | 'description';

export type BuildInputFieldArg = {
  schemaObject: InvocationFieldSchema;
  baseField: Omit<InputFieldTemplateBase, 'type'>;
};

/**
 * Transforms an invocation output ref object to field type.
 * @param ref The ref string to transform
 * @returns The field type.
 *
 * @example
 * refObjectToFieldType({ "$ref": "#/components/schemas/ImageField" }) --> 'ImageField'
 */
export const refObjectToFieldType = (
  refObject: OpenAPIV3.ReferenceObject
): FieldType => {
  const name = refObject.$ref.split('/').slice(-1)[0];
  if (!name) {
    throw `Unknown field type: ${name}`;
  }
  return name as FieldType;
};

const buildIntegerInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): IntegerInputFieldTemplate => {
  const template: IntegerInputFieldTemplate = {
    ...baseField,
    type: 'integer',
    default: schemaObject.default ?? 0,
  };

  if (schemaObject.multipleOf !== undefined) {
    template.multipleOf = schemaObject.multipleOf;
  }

  if (schemaObject.maximum !== undefined) {
    template.maximum = schemaObject.maximum;
  }

  if (schemaObject.exclusiveMaximum !== undefined) {
    template.exclusiveMaximum = schemaObject.exclusiveMaximum;
  }

  if (schemaObject.minimum !== undefined) {
    template.minimum = schemaObject.minimum;
  }

  if (schemaObject.exclusiveMinimum !== undefined) {
    template.exclusiveMinimum = schemaObject.exclusiveMinimum;
  }

  return template;
};

const buildFloatInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): FloatInputFieldTemplate => {
  const template: FloatInputFieldTemplate = {
    ...baseField,
    type: 'float',
    default: schemaObject.default ?? 0,
  };

  if (schemaObject.multipleOf !== undefined) {
    template.multipleOf = schemaObject.multipleOf;
  }

  if (schemaObject.maximum !== undefined) {
    template.maximum = schemaObject.maximum;
  }

  if (schemaObject.exclusiveMaximum !== undefined) {
    template.exclusiveMaximum = schemaObject.exclusiveMaximum;
  }

  if (schemaObject.minimum !== undefined) {
    template.minimum = schemaObject.minimum;
  }

  if (schemaObject.exclusiveMinimum !== undefined) {
    template.exclusiveMinimum = schemaObject.exclusiveMinimum;
  }

  return template;
};

const buildStringInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): StringInputFieldTemplate => {
  const template: StringInputFieldTemplate = {
    ...baseField,
    type: 'string',
    default: schemaObject.default ?? '',
  };

  if (schemaObject.minLength !== undefined) {
    template.minLength = schemaObject.minLength;
  }

  if (schemaObject.maxLength !== undefined) {
    template.maxLength = schemaObject.maxLength;
  }

  if (schemaObject.pattern !== undefined) {
    template.pattern = schemaObject.pattern;
  }

  return template;
};

const buildBooleanInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): BooleanInputFieldTemplate => {
  const template: BooleanInputFieldTemplate = {
    ...baseField,
    type: 'boolean',
    default: schemaObject.default ?? false,
  };

  return template;
};

const buildMainModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): MainModelInputFieldTemplate => {
  const template: MainModelInputFieldTemplate = {
    ...baseField,
    type: 'MainModelField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildSDXLMainModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): SDXLMainModelInputFieldTemplate => {
  const template: SDXLMainModelInputFieldTemplate = {
    ...baseField,
    type: 'SDXLMainModelField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildRefinerModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): SDXLRefinerModelInputFieldTemplate => {
  const template: SDXLRefinerModelInputFieldTemplate = {
    ...baseField,
    type: 'SDXLRefinerModelField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildVaeModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): VaeModelInputFieldTemplate => {
  const template: VaeModelInputFieldTemplate = {
    ...baseField,
    type: 'VaeModelField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildLoRAModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): LoRAModelInputFieldTemplate => {
  const template: LoRAModelInputFieldTemplate = {
    ...baseField,
    type: 'LoRAModelField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildControlNetModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ControlNetModelInputFieldTemplate => {
  const template: ControlNetModelInputFieldTemplate = {
    ...baseField,
    type: 'ControlNetModelField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildImageInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ImageInputFieldTemplate => {
  const template: ImageInputFieldTemplate = {
    ...baseField,
    type: 'ImageField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildImageCollectionInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ImageCollectionInputFieldTemplate => {
  const template: ImageCollectionInputFieldTemplate = {
    ...baseField,
    type: 'ImageCollection',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildDenoiseMaskInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): DenoiseMaskInputFieldTemplate => {
  const template: DenoiseMaskInputFieldTemplate = {
    ...baseField,
    type: 'DenoiseMaskField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildLatentsInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): LatentsInputFieldTemplate => {
  const template: LatentsInputFieldTemplate = {
    ...baseField,
    type: 'LatentsField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildConditioningInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ConditioningInputFieldTemplate => {
  const template: ConditioningInputFieldTemplate = {
    ...baseField,
    type: 'ConditioningField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildUNetInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): UNetInputFieldTemplate => {
  const template: UNetInputFieldTemplate = {
    ...baseField,
    type: 'UNetField',

    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildClipInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ClipInputFieldTemplate => {
  const template: ClipInputFieldTemplate = {
    ...baseField,
    type: 'ClipField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildVaeInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): VaeInputFieldTemplate => {
  const template: VaeInputFieldTemplate = {
    ...baseField,
    type: 'VaeField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildControlInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ControlInputFieldTemplate => {
  const template: ControlInputFieldTemplate = {
    ...baseField,
    type: 'ControlField',
    default: schemaObject.default ?? undefined,
  };

  return template;
};

const buildEnumInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): EnumInputFieldTemplate => {
  const options = schemaObject.enum ?? [];
  const template: EnumInputFieldTemplate = {
    ...baseField,
    type: 'enum',
    enumType: (schemaObject.type as 'string' | 'number') ?? 'string', // TODO: dangerous?
    options: options,
    default: schemaObject.default ?? options[0],
  };

  return template;
};

const buildCollectionInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): CollectionInputFieldTemplate => {
  const template: CollectionInputFieldTemplate = {
    ...baseField,
    type: 'Collection',
    default: [],
  };

  return template;
};

const buildCollectionItemInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): CollectionItemInputFieldTemplate => {
  const template: CollectionItemInputFieldTemplate = {
    ...baseField,
    type: 'CollectionItem',
    default: undefined,
  };

  return template;
};

const buildColorInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ColorInputFieldTemplate => {
  const template: ColorInputFieldTemplate = {
    ...baseField,
    type: 'ColorField',
    default: schemaObject.default ?? { r: 127, g: 127, b: 127, a: 255 },
  };

  return template;
};

const buildSchedulerInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): SchedulerInputFieldTemplate => {
  const template: SchedulerInputFieldTemplate = {
    ...baseField,
    type: 'Scheduler',
    default: schemaObject.default ?? 'euler',
  };

  return template;
};

export const getFieldType = (schemaObject: InvocationFieldSchema): string => {
  let fieldType = '';

  const { ui_type } = schemaObject;
  if (ui_type) {
    fieldType = ui_type;
  } else if (!schemaObject.type) {
    // console.log('refObject', schemaObject);
    // if schemaObject has no type, then it should have one of allOf, anyOf, oneOf
    if (schemaObject.allOf) {
      fieldType = refObjectToFieldType(
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        schemaObject.allOf![0] as OpenAPIV3.ReferenceObject
      );
    } else if (schemaObject.anyOf) {
      fieldType = refObjectToFieldType(
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        schemaObject.anyOf![0] as OpenAPIV3.ReferenceObject
      );
    } else if (schemaObject.oneOf) {
      fieldType = refObjectToFieldType(
        // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
        schemaObject.oneOf![0] as OpenAPIV3.ReferenceObject
      );
    }
  } else if (schemaObject.enum) {
    fieldType = 'enum';
  } else if (schemaObject.type) {
    if (schemaObject.type === 'number') {
      // floats are "number" in OpenAPI, while ints are "integer"
      fieldType = 'float';
    } else {
      fieldType = schemaObject.type;
    }
  }

  return fieldType;
};

/**
 * Builds an input field from an invocation schema property.
 * @param fieldSchema The schema object
 * @returns An input field
 */
export const buildInputFieldTemplate = (
  nodeSchema: InvocationSchemaObject,
  fieldSchema: InvocationFieldSchema,
  name: string,
  fieldType: FieldType
) => {
  const { input, ui_hidden, ui_component, ui_type, ui_order } = fieldSchema;

  const extra = {
    input,
    ui_hidden,
    ui_component,
    ui_type,
    required: nodeSchema.required?.includes(name) ?? false,
    ui_order,
  };

  const baseField = {
    name,
    title: fieldSchema.title ?? '',
    description: fieldSchema.description ?? '',
    fieldKind: 'input' as const,
    ...extra,
  };

  if (fieldType === 'ImageField') {
    return buildImageInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'ImageCollection') {
    return buildImageCollectionInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'DenoiseMaskField') {
    return buildDenoiseMaskInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'LatentsField') {
    return buildLatentsInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'ConditioningField') {
    return buildConditioningInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'UNetField') {
    return buildUNetInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'ClipField') {
    return buildClipInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'VaeField') {
    return buildVaeInputFieldTemplate({ schemaObject: fieldSchema, baseField });
  }
  if (fieldType === 'ControlField') {
    return buildControlInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'MainModelField') {
    return buildMainModelInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'SDXLRefinerModelField') {
    return buildRefinerModelInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'SDXLMainModelField') {
    return buildSDXLMainModelInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'VaeModelField') {
    return buildVaeModelInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'LoRAModelField') {
    return buildLoRAModelInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'ControlNetModelField') {
    return buildControlNetModelInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'enum') {
    return buildEnumInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'integer') {
    return buildIntegerInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'float') {
    return buildFloatInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'string') {
    return buildStringInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'boolean') {
    return buildBooleanInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'Collection') {
    return buildCollectionInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'CollectionItem') {
    return buildCollectionItemInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'ColorField') {
    return buildColorInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  if (fieldType === 'Scheduler') {
    return buildSchedulerInputFieldTemplate({
      schemaObject: fieldSchema,
      baseField,
    });
  }
  return;
};
