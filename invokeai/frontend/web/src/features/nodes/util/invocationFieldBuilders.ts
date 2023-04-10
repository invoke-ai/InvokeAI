import { reduce } from 'lodash';
import { OpenAPIV3 } from 'openapi-types';
import { FIELD_TYPE_MAP } from '../constants';
import {
  BooleanInputField,
  EnumInputField,
  FloatInputField,
  ImageInputField,
  IntegerInputField,
  LatentsInputField,
  OutputField,
  StringInputField,
  isSchemaObject,
  ModelInputField,
  TypeHints,
  FieldType,
  isReferenceObject,
} from '../types';

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

const buildIntegerInputField = (
  input: OpenAPIV3.SchemaObject,
  name: string
): IntegerInputField => {
  const field: IntegerInputField = {
    type: 'integer',
    name,
    title: input.title ?? '',
    description: input.description ?? '',
    value: input.default ?? 0,
  };

  if (input.multipleOf !== undefined) {
    field.multipleOf = input.multipleOf;
  }

  if (input.maximum !== undefined) {
    field.maximum = input.maximum;
  }

  if (input.exclusiveMaximum !== undefined) {
    field.exclusiveMaximum = input.exclusiveMaximum;
  }

  if (input.minimum !== undefined) {
    field.minimum = input.minimum;
  }

  if (input.exclusiveMinimum !== undefined) {
    field.exclusiveMinimum = input.exclusiveMinimum;
  }

  return field;
};

const buildFloatInputField = (
  input: OpenAPIV3.SchemaObject,
  name: string
): FloatInputField => {
  const field: FloatInputField = {
    type: 'float',
    name,
    title: input.title ?? '',
    description: input.description ?? '',
    value: input.default ?? 0,
  };

  if (input.multipleOf !== undefined) {
    field.multipleOf = input.multipleOf;
  }

  if (input.maximum !== undefined) {
    field.maximum = input.maximum;
  }

  if (input.exclusiveMaximum !== undefined) {
    field.exclusiveMaximum = input.exclusiveMaximum;
  }

  if (input.minimum !== undefined) {
    field.minimum = input.minimum;
  }

  if (input.exclusiveMinimum !== undefined) {
    field.exclusiveMinimum = input.exclusiveMinimum;
  }

  return field;
};

const buildStringInputField = (
  input: OpenAPIV3.SchemaObject,
  name: string
): StringInputField => {
  const field: StringInputField = {
    type: 'string',
    name,
    title: input.title ?? '',
    description: input.description ?? '',
    value: input.default ?? '',
  };

  if (input.minLength !== undefined) {
    field.minLength = input.minLength;
  }

  if (input.maxLength !== undefined) {
    field.maxLength = input.maxLength;
  }

  if (input.pattern !== undefined) {
    field.pattern = input.pattern;
  }

  return field;
};

const buildBooleanInputField = (
  input: OpenAPIV3.SchemaObject,
  name: string
): BooleanInputField => {
  const field: BooleanInputField = {
    type: 'boolean',
    name,
    title: input.title ?? '',
    description: input.description ?? '',
    value: input.default ?? false,
  };

  return field;
};

const buildModelInputField = (
  input: OpenAPIV3.SchemaObject,
  name: string
): ModelInputField => {
  const field: ModelInputField = {
    type: 'model',
    name,
    title: input.title ?? '',
    description: input.description ?? '',
    value: input.default ?? '',
  };

  return field;
};

const buildImageInputField = (
  input: OpenAPIV3.SchemaObject,
  name: string
): ImageInputField => {
  const field: ImageInputField = {
    type: 'image',
    name,
    title: input.title ?? '',
    description: input.description ?? '',
    value: input.default ?? '',
  };

  return field;
};

const buildLatentsInputField = (
  input: OpenAPIV3.SchemaObject,
  name: string
): LatentsInputField => {
  const field: LatentsInputField = {
    type: 'latents',
    name,
    title: input.title ?? '',
    description: input.description ?? '',
    value: input.default ?? '',
  };

  return field;
};

const buildEnumInputField = (
  input: OpenAPIV3.SchemaObject,
  name: string
): EnumInputField => {
  const field: EnumInputField = {
    type: 'enum',
    name,
    title: input.title ?? '',
    value: input.default,
    enumType: (input.type as 'string' | 'number') ?? 'string', // TODO: dangerous?
    options: input.enum ?? [],
    description: input.description ?? '',
  };

  return field;
};

export const getFieldType = (
  schemaObject: OpenAPIV3.SchemaObject,
  name: string,
  typeHints?: TypeHints
): FieldType | undefined => {
  let rawFieldType = '';

  if (typeHints && name in typeHints) {
    rawFieldType = typeHints[name];
  } else if (!schemaObject.type) {
    rawFieldType = refObjectToFieldType(
      schemaObject.allOf![0] as OpenAPIV3.ReferenceObject
    );
  } else if (schemaObject.enum) {
    rawFieldType = 'enum';
  } else if (schemaObject.type) {
    rawFieldType = schemaObject.type;
  }

  return FIELD_TYPE_MAP[rawFieldType];
};

/**
 * Builds an input field from an invocation schema property.
 * @param schemaObject The schema object
 * @returns An input field
 */
export const buildInputField = (
  schemaObject: OpenAPIV3.SchemaObject,
  name: string,
  typeHints?: TypeHints
) => {
  const fieldType = getFieldType(schemaObject, name, typeHints);

  if (!fieldType) {
    throw `Field type "${fieldType}" is unknown!`;
  }

  if (['image', 'ImageField'].includes(fieldType)) {
    return buildImageInputField(schemaObject, name);
  }
  if (['latents', 'LatentsField'].includes(fieldType)) {
    return buildLatentsInputField(schemaObject, name);
  }
  if (['model'].includes(fieldType)) {
    return buildModelInputField(schemaObject, name);
  }
  if (['enum'].includes(fieldType)) {
    return buildEnumInputField(schemaObject, name);
  }
  if (['integer'].includes(fieldType)) {
    return buildIntegerInputField(schemaObject, name);
  }
  if (['number', 'float'].includes(fieldType)) {
    return buildFloatInputField(schemaObject, name);
  }
  if (['string'].includes(fieldType)) {
    return buildStringInputField(schemaObject, name);
  }
  if (['boolean'].includes(fieldType)) {
    return buildBooleanInputField(schemaObject, name);
  }

  return;
};

/**
 * Builds invocation output fields from an invocation's output reference object.
 * @param openAPI The OpenAPI schema
 * @param refObject The output reference object
 * @returns A record of outputs
 */
export const buildOutputFields = (
  refObject: OpenAPIV3.ReferenceObject,
  openAPI: OpenAPIV3.Document,
  typeHints?: TypeHints
): Record<string, OutputField> => {
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
          isSchemaObject(property)
        ) {
          const fieldType = getFieldType(property, propertyName, typeHints);

          if (!fieldType) {
            throw `Field type "${fieldType}" is unknown!`;
          }

          outputsAccumulator[propertyName] = {
            name: propertyName,
            title: property.title ?? '',
            description: property.description ?? '',
            type: fieldType,
          };
        }

        return outputsAccumulator;
      },
      {} as Record<string, OutputField>
    );

    return outputFields;
  }

  return {};
};
