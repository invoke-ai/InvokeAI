import { reduce } from 'lodash-es';
import { OpenAPIV3 } from 'openapi-types';
import { FIELD_TYPE_MAP } from '../types/constants';
import { isSchemaObject } from '../types/typeGuards';
import {
  BooleanInputFieldTemplate,
  EnumInputFieldTemplate,
  FloatInputFieldTemplate,
  ImageInputFieldTemplate,
  IntegerInputFieldTemplate,
  LatentsInputFieldTemplate,
  ConditioningInputFieldTemplate,
  ControlInputFieldTemplate,
  StringInputFieldTemplate,
  ModelInputFieldTemplate,
  ArrayInputFieldTemplate,
  ItemInputFieldTemplate,
  ColorInputFieldTemplate,
  InputFieldTemplateBase,
  OutputFieldTemplate,
  TypeHints,
  FieldType,
} from '../types/types';

export type BaseFieldProperties = 'name' | 'title' | 'description';

export type BuildInputFieldArg = {
  schemaObject: OpenAPIV3.SchemaObject;
  baseField: Omit<
    InputFieldTemplateBase,
    'type' | 'inputRequirement' | 'inputKind'
  >;
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
): keyof typeof FIELD_TYPE_MAP => refObject.$ref.split('/').slice(-1)[0];

const buildIntegerInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): IntegerInputFieldTemplate => {
  const template: IntegerInputFieldTemplate = {
    ...baseField,
    type: 'integer',
    inputRequirement: 'always',
    inputKind: 'any',
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
    inputRequirement: 'always',
    inputKind: 'any',
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
    inputRequirement: 'always',
    inputKind: 'any',
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
    inputRequirement: 'always',
    inputKind: 'any',
    default: schemaObject.default ?? false,
  };

  return template;
};

const buildModelInputFieldTemplate = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ModelInputFieldTemplate => {
  const template: ModelInputFieldTemplate = {
    ...baseField,
    type: 'model',
    inputRequirement: 'always',
    inputKind: 'direct',
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
    type: 'image',
    inputRequirement: 'always',
    inputKind: 'any',
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
    type: 'latents',
    inputRequirement: 'always',
    inputKind: 'connection',
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
    type: 'conditioning',
    inputRequirement: 'always',
    inputKind: 'connection',
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
    type: 'control',
    inputRequirement: 'always',
    inputKind: 'connection',
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
    inputRequirement: 'always',
    inputKind: 'direct',
    default: schemaObject.default ?? options[0],
  };

  return template;
};

const buildArrayInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): ArrayInputFieldTemplate => {
  const template: ArrayInputFieldTemplate = {
    ...baseField,
    type: 'array',
    inputRequirement: 'always',
    inputKind: 'direct',
    default: [],
  };

  return template;
};

const buildItemInputFieldTemplate = ({
  baseField,
}: BuildInputFieldArg): ItemInputFieldTemplate => {
  const template: ItemInputFieldTemplate = {
    ...baseField,
    type: 'item',
    inputRequirement: 'always',
    inputKind: 'direct',
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
    type: 'color',
    inputRequirement: 'always',
    inputKind: 'direct',
    default: schemaObject.default ?? { r: 127, g: 127, b: 127, a: 255 },
  };

  return template;
};

export const getFieldType = (
  schemaObject: OpenAPIV3.SchemaObject,
  name: string,
  typeHints?: TypeHints
): FieldType => {
  let rawFieldType = '';

  if (typeHints && name in typeHints) {
    rawFieldType = typeHints[name];
  } else if (!schemaObject.type) {
    // if schemaObject has no type, then it should have one of allOf, anyOf, oneOf
    if (schemaObject.allOf) {
      rawFieldType = refObjectToFieldType(
        schemaObject.allOf![0] as OpenAPIV3.ReferenceObject
      );
    } else if (schemaObject.anyOf) {
      rawFieldType = refObjectToFieldType(
        schemaObject.anyOf![0] as OpenAPIV3.ReferenceObject
      );
    } else if (schemaObject.oneOf) {
      rawFieldType = refObjectToFieldType(
        schemaObject.oneOf![0] as OpenAPIV3.ReferenceObject
      );
    }
  } else if (schemaObject.enum) {
    rawFieldType = 'enum';
  } else if (schemaObject.type) {
    rawFieldType = schemaObject.type;
  }

  const fieldType = FIELD_TYPE_MAP[rawFieldType];

  if (!fieldType) {
    throw `Field type "${rawFieldType}" is unknown!`;
  }

  return fieldType;
};

/**
 * Builds an input field from an invocation schema property.
 * @param schemaObject The schema object
 * @returns An input field
 */
export const buildInputFieldTemplate = (
  schemaObject: OpenAPIV3.SchemaObject,
  name: string,
  typeHints?: TypeHints
) => {
  const fieldType = getFieldType(schemaObject, name, typeHints);

  const baseField = {
    name,
    title: schemaObject.title ?? '',
    description: schemaObject.description ?? '',
  };

  if (['image'].includes(fieldType)) {
    return buildImageInputFieldTemplate({ schemaObject, baseField });
  }
  if (['latents'].includes(fieldType)) {
    return buildLatentsInputFieldTemplate({ schemaObject, baseField });
  }
  if (['conditioning'].includes(fieldType)) {
    return buildConditioningInputFieldTemplate({ schemaObject, baseField });
  }
  if (['control'].includes(fieldType)) {
    return buildControlInputFieldTemplate({ schemaObject, baseField });
  }
  if (['model'].includes(fieldType)) {
    return buildModelInputFieldTemplate({ schemaObject, baseField });
  }
  if (['enum'].includes(fieldType)) {
    return buildEnumInputFieldTemplate({ schemaObject, baseField });
  }
  if (['integer'].includes(fieldType)) {
    return buildIntegerInputFieldTemplate({ schemaObject, baseField });
  }
  if (['number', 'float'].includes(fieldType)) {
    return buildFloatInputFieldTemplate({ schemaObject, baseField });
  }
  if (['string'].includes(fieldType)) {
    return buildStringInputFieldTemplate({ schemaObject, baseField });
  }
  if (['boolean'].includes(fieldType)) {
    return buildBooleanInputFieldTemplate({ schemaObject, baseField });
  }
  if (['array'].includes(fieldType)) {
    return buildArrayInputFieldTemplate({ schemaObject, baseField });
  }
  if (['item'].includes(fieldType)) {
    return buildItemInputFieldTemplate({ schemaObject, baseField });
  }
  if (['color'].includes(fieldType)) {
    return buildColorInputFieldTemplate({ schemaObject, baseField });
  }
  if (['array'].includes(fieldType)) {
    return buildArrayInputFieldTemplate({ schemaObject, baseField });
  }
  if (['item'].includes(fieldType)) {
    return buildItemInputFieldTemplate({ schemaObject, baseField });
  }

  return;
};

/**
 * Builds invocation output fields from an invocation's output reference object.
 * @param openAPI The OpenAPI schema
 * @param refObject The output reference object
 * @returns A record of outputs
 */
export const buildOutputFieldTemplates = (
  refObject: OpenAPIV3.ReferenceObject,
  openAPI: OpenAPIV3.Document,
  typeHints?: TypeHints
): Record<string, OutputFieldTemplate> => {
  // extract output schema name from ref
  const outputSchemaName = refObject.$ref.split('/').slice(-1)[0];

  // get the output schema itself
  const outputSchema = openAPI.components!.schemas![outputSchemaName];

  if (isSchemaObject(outputSchema)) {
    const outputFields = reduce(
      outputSchema.properties as OpenAPIV3.SchemaObject,
      (outputsAccumulator, property, propertyName) => {
        if (
          !['type', 'id'].includes(propertyName) &&
          !['object'].includes(property.type) && // TODO: handle objects?
          isSchemaObject(property)
        ) {
          const fieldType = getFieldType(property, propertyName, typeHints);

          outputsAccumulator[propertyName] = {
            name: propertyName,
            title: property.title ?? '',
            description: property.description ?? '',
            type: fieldType,
          };
        }

        return outputsAccumulator;
      },
      {} as Record<string, OutputFieldTemplate>
    );

    return outputFields;
  }

  return {};
};
