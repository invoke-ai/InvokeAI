import { filter } from 'lodash';
import { OpenAPIV3 } from 'openapi-types';
import {
  BooleanInputField,
  EnumInputField,
  FIELD_TYPE_MAP,
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
  input: OpenAPIV3.SchemaObject
): IntegerInputField => {
  const field: IntegerInputField = {
    type: 'integer',
    title: input.title ?? '',
    description: input.description ?? '',
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
  input: OpenAPIV3.SchemaObject
): FloatInputField => {
  const field: FloatInputField = {
    type: 'float',
    title: input.title ?? '',
    description: input.description ?? '',
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
  input: OpenAPIV3.SchemaObject
): StringInputField => {
  const field: StringInputField = {
    type: 'string',
    title: input.title ?? '',
    description: input.description ?? '',
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
  input: OpenAPIV3.SchemaObject
): BooleanInputField => {
  const field: BooleanInputField = {
    type: 'boolean',
    title: input.title ?? '',
    description: input.description ?? '',
  };

  return field;
};

const buildImageInputField = (
  input: OpenAPIV3.SchemaObject
): ImageInputField => {
  const field: ImageInputField = {
    type: 'image',
    title: input.title ?? '',
    description: input.description ?? '',
  };

  return field;
};

const buildLatentsInputField = (
  input: OpenAPIV3.SchemaObject
): LatentsInputField => {
  const field: LatentsInputField = {
    type: 'latents',
    title: input.title ?? '',
    description: input.description ?? '',
  };

  return field;
};

const buildEnumInputField = (input: OpenAPIV3.SchemaObject): EnumInputField => {
  const field: EnumInputField = {
    type: 'enum',
    title: input.title ?? '',
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
export const buildInputField = (schemaObject: OpenAPIV3.SchemaObject) => {
  if (!schemaObject.type) {
    // the this input/output is a ref! extract the ref string
    const rawType = refObjectToFieldType(
      schemaObject.allOf![0] as OpenAPIV3.ReferenceObject
    );

    const fieldType = FIELD_TYPE_MAP[rawType];

    if (fieldType === 'image') {
      return buildImageInputField(schemaObject);
    }
    if (fieldType === 'latents') {
      return buildLatentsInputField(schemaObject);
    }
  }
  if (schemaObject.enum) {
    return buildEnumInputField(schemaObject);
  }
  if (schemaObject.type === 'integer') {
    return buildIntegerInputField(schemaObject);
  }
  if (schemaObject.type === 'number') {
    return buildFloatInputField(schemaObject);
  }
  if (schemaObject.type === 'string') {
    return buildStringInputField(schemaObject);
  }
  if (schemaObject.type === 'boolean') {
    return buildBooleanInputField(schemaObject);
  }

  return;
};

/**
 * Builds invocation output fields from an invocation's output reference object.
 * @param openAPI The OpenAPI schema
 * @param refObject The output reference object
 * @returns An array of outputs
 */
export const buildOutputFields = (
  refObject: OpenAPIV3.ReferenceObject,
  openAPI: OpenAPIV3.Document
): OutputField[] => {
  // extract output schema name from ref
  const outputSchemaName = refObject.$ref.split('/').slice(-1)[0];

  // get the output schema itself
  const outputSchema = openAPI.components!.schemas![outputSchemaName];

  // filter out 'type' properties of the schema
  const filteredProperties = filter(
    (outputSchema as OpenAPIV3.SchemaObject).properties,
    (prop, key) => key !== 'type'
  );

  const outputFields: OutputField[] = [];

  filteredProperties.forEach((property) => {
    if (isSchemaObject(property)) {
      // we need to parse the ref to get the actual output field types
      let rawType: string;

      if (property.allOf) {
        rawType = refObjectToFieldType(
          property.allOf[0] as OpenAPIV3.ReferenceObject
        );
      } else {
        // we can just use the property's type
        rawType = property.type!;
      }

      outputFields.push({
        title: property.title ?? '',
        description: property.description ?? '',
        type: FIELD_TYPE_MAP[rawType],
      });
    }
  });

  return outputFields;
};
