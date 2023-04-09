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

/**
 * Builds an input field from an invocation schema property.
 * @param schemaObject The schema object
 * @returns An input field
 */
export const buildInputField = (
  schemaObject: OpenAPIV3.SchemaObject,
  name: string
) => {
  if (!schemaObject.type) {
    // the this input/output is a ref! extract the ref string
    const rawType = refObjectToFieldType(
      schemaObject.allOf![0] as OpenAPIV3.ReferenceObject
    );

    const fieldType = FIELD_TYPE_MAP[rawType];

    if (fieldType === 'image') {
      return buildImageInputField(schemaObject, name);
    }
    if (fieldType === 'latents') {
      return buildLatentsInputField(schemaObject, name);
    }
  }
  if (schemaObject.enum) {
    return buildEnumInputField(schemaObject, name);
  }
  if (schemaObject.type === 'integer') {
    return buildIntegerInputField(schemaObject, name);
  }
  if (schemaObject.type === 'number') {
    return buildFloatInputField(schemaObject, name);
  }
  if (schemaObject.type === 'string') {
    return buildStringInputField(schemaObject, name);
  }
  if (schemaObject.type === 'boolean') {
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
  openAPI: OpenAPIV3.Document
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
          let rawType: string;

          if (property.allOf) {
            // we need to parse the ref to get the actual output field types
            rawType = refObjectToFieldType(
              property.allOf[0] as OpenAPIV3.ReferenceObject
            );
          } else {
            // we can just use the property's type
            rawType = property.type!;
          }

          outputsAccumulator[propertyName] = {
            name: propertyName,
            title: property.title ?? '',
            description: property.description ?? '',
            type: FIELD_TYPE_MAP[rawType],
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
