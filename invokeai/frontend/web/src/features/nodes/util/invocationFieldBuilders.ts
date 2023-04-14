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
  InputField,
} from '../types';

export type BaseFieldProperties = 'name' | 'title' | 'description';

export type BuildInputFieldArg = {
  schemaObject: OpenAPIV3.SchemaObject;
  baseField: Pick<InputField, BaseFieldProperties>;
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

const buildIntegerInputField = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): IntegerInputField => {
  const field: Omit<IntegerInputField, BaseFieldProperties> = {
    type: 'integer',
    value: schemaObject.default ?? 0,
  };

  if (schemaObject.multipleOf !== undefined) {
    field.multipleOf = schemaObject.multipleOf;
  }

  if (schemaObject.maximum !== undefined) {
    field.maximum = schemaObject.maximum;
  }

  if (schemaObject.exclusiveMaximum !== undefined) {
    field.exclusiveMaximum = schemaObject.exclusiveMaximum;
  }

  if (schemaObject.minimum !== undefined) {
    field.minimum = schemaObject.minimum;
  }

  if (schemaObject.exclusiveMinimum !== undefined) {
    field.exclusiveMinimum = schemaObject.exclusiveMinimum;
  }

  return { ...baseField, ...field };
};

const buildFloatInputField = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): FloatInputField => {
  const field: Omit<FloatInputField, BaseFieldProperties> = {
    type: 'float',
    value: schemaObject.default ?? 0,
  };

  if (schemaObject.multipleOf !== undefined) {
    field.multipleOf = schemaObject.multipleOf;
  }

  if (schemaObject.maximum !== undefined) {
    field.maximum = schemaObject.maximum;
  }

  if (schemaObject.exclusiveMaximum !== undefined) {
    field.exclusiveMaximum = schemaObject.exclusiveMaximum;
  }

  if (schemaObject.minimum !== undefined) {
    field.minimum = schemaObject.minimum;
  }

  if (schemaObject.exclusiveMinimum !== undefined) {
    field.exclusiveMinimum = schemaObject.exclusiveMinimum;
  }

  return { ...baseField, ...field };
};

const buildStringInputField = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): StringInputField => {
  const field: Omit<StringInputField, BaseFieldProperties> = {
    type: 'string',
    value: schemaObject.default ?? '',
  };

  if (schemaObject.minLength !== undefined) {
    field.minLength = schemaObject.minLength;
  }

  if (schemaObject.maxLength !== undefined) {
    field.maxLength = schemaObject.maxLength;
  }

  if (schemaObject.pattern !== undefined) {
    field.pattern = schemaObject.pattern;
  }

  return { ...baseField, ...field };
};

const buildBooleanInputField = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): BooleanInputField => {
  const field: Omit<BooleanInputField, BaseFieldProperties> = {
    type: 'boolean',
    value: schemaObject.default ?? false,
  };

  return { ...baseField, ...field };
};

const buildModelInputField = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ModelInputField => {
  const field: Omit<ModelInputField, BaseFieldProperties> = {
    type: 'model',
    value: schemaObject.default ?? '',
    connectionType: 'never',
  };

  return { ...baseField, ...field };
};

const buildImageInputField = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): ImageInputField => {
  const field: Omit<ImageInputField, BaseFieldProperties> = {
    type: 'image',
    value: schemaObject.default ?? '',
    connectionType: 'always',
  };

  return { ...baseField, ...field };
};

const buildLatentsInputField = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): LatentsInputField => {
  const field: Omit<LatentsInputField, BaseFieldProperties> = {
    type: 'latents',
    value: schemaObject.default ?? '',
    connectionType: 'always',
  };

  return { ...baseField, ...field };
};

const buildEnumInputField = ({
  schemaObject,
  baseField,
}: BuildInputFieldArg): EnumInputField => {
  const field: Omit<EnumInputField, BaseFieldProperties> = {
    ...baseField,
    type: 'enum',
    value: schemaObject.default,
    enumType: (schemaObject.type as 'string' | 'number') ?? 'string', // TODO: dangerous?
    options: schemaObject.enum ?? [],
  };

  return { ...baseField, ...field };
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
    rawFieldType = refObjectToFieldType(
      schemaObject.allOf![0] as OpenAPIV3.ReferenceObject
    );
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
export const buildInputField = (
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
    return buildImageInputField({ schemaObject, baseField });
  }
  if (['latents'].includes(fieldType)) {
    return buildLatentsInputField({ schemaObject, baseField });
  }
  if (['model'].includes(fieldType)) {
    return buildModelInputField({ schemaObject, baseField });
  }
  if (['enum'].includes(fieldType)) {
    return buildEnumInputField({ schemaObject, baseField });
  }
  if (['integer'].includes(fieldType)) {
    return buildIntegerInputField({ schemaObject, baseField });
  }
  if (['number', 'float'].includes(fieldType)) {
    return buildFloatInputField({ schemaObject, baseField });
  }
  if (['string'].includes(fieldType)) {
    return buildStringInputField({ schemaObject, baseField });
  }
  if (['boolean'].includes(fieldType)) {
    return buildBooleanInputField({ schemaObject, baseField });
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
