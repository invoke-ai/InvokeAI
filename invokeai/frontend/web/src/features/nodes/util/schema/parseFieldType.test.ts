import {
  UnableToExtractSchemaNameFromRefError,
  UnsupportedArrayItemType,
  UnsupportedPrimitiveTypeError,
  UnsupportedUnionError,
} from 'features/nodes/types/error';
import type { FieldType } from 'features/nodes/types/field';
import type { InvocationFieldSchema, OpenAPIV3_1SchemaOrRef } from 'features/nodes/types/openapi';
import { parseFieldType, refObjectToSchemaName } from 'features/nodes/util/schema/parseFieldType';
import { describe, expect, it } from 'vitest';

type ParseFieldTypeTestCase = {
  name: string;
  schema: OpenAPIV3_1SchemaOrRef | InvocationFieldSchema;
  expected: FieldType;
};

const primitiveTypes: ParseFieldTypeTestCase[] = [
  {
    name: 'SINGLE IntegerField',
    schema: { type: 'integer' },
    expected: { name: 'IntegerField', cardinality: 'SINGLE' },
  },
  {
    name: 'SINGLE FloatField',
    schema: { type: 'number' },
    expected: { name: 'FloatField', cardinality: 'SINGLE' },
  },
  {
    name: 'SINGLE StringField',
    schema: { type: 'string' },
    expected: { name: 'StringField', cardinality: 'SINGLE' },
  },
  {
    name: 'SINGLE BooleanField',
    schema: { type: 'boolean' },
    expected: { name: 'BooleanField', cardinality: 'SINGLE' },
  },
  {
    name: 'COLLECTION IntegerField',
    schema: { items: { type: 'integer' }, type: 'array' },
    expected: { name: 'IntegerField', cardinality: 'COLLECTION' },
  },
  {
    name: 'COLLECTION FloatField',
    schema: { items: { type: 'number' }, type: 'array' },
    expected: { name: 'FloatField', cardinality: 'COLLECTION' },
  },
  {
    name: 'COLLECTION StringField',
    schema: { items: { type: 'string' }, type: 'array' },
    expected: { name: 'StringField', cardinality: 'COLLECTION' },
  },
  {
    name: 'COLLECTION BooleanField',
    schema: { items: { type: 'boolean' }, type: 'array' },
    expected: { name: 'BooleanField', cardinality: 'COLLECTION' },
  },
  {
    name: 'SINGLE_OR_COLLECTION IntegerField',
    schema: {
      anyOf: [
        {
          type: 'integer',
        },
        {
          items: {
            type: 'integer',
          },
          type: 'array',
        },
      ],
    },
    expected: { name: 'IntegerField', cardinality: 'SINGLE_OR_COLLECTION' },
  },
  {
    name: 'SINGLE_OR_COLLECTION FloatField',
    schema: {
      anyOf: [
        {
          type: 'number',
        },
        {
          items: {
            type: 'number',
          },
          type: 'array',
        },
      ],
    },
    expected: { name: 'FloatField', cardinality: 'SINGLE_OR_COLLECTION' },
  },
  {
    name: 'SINGLE_OR_COLLECTION StringField',
    schema: {
      anyOf: [
        {
          type: 'string',
        },
        {
          items: {
            type: 'string',
          },
          type: 'array',
        },
      ],
    },
    expected: { name: 'StringField', cardinality: 'SINGLE_OR_COLLECTION' },
  },
  {
    name: 'SINGLE_OR_COLLECTION BooleanField',
    schema: {
      anyOf: [
        {
          type: 'boolean',
        },
        {
          items: {
            type: 'boolean',
          },
          type: 'array',
        },
      ],
    },
    expected: { name: 'BooleanField', cardinality: 'SINGLE_OR_COLLECTION' },
  },
];

const complexTypes: ParseFieldTypeTestCase[] = [
  {
    name: 'SINGLE ConditioningField',
    schema: {
      allOf: [
        {
          $ref: '#/components/schemas/ConditioningField',
        },
      ],
    },
    expected: { name: 'ConditioningField', cardinality: 'SINGLE' },
  },
  {
    name: 'Nullable SINGLE ConditioningField',
    schema: {
      anyOf: [
        {
          $ref: '#/components/schemas/ConditioningField',
        },
        {
          type: 'null',
        },
      ],
    },
    expected: { name: 'ConditioningField', cardinality: 'SINGLE' },
  },
  {
    name: 'COLLECTION ConditioningField',
    schema: {
      anyOf: [
        {
          items: {
            $ref: '#/components/schemas/ConditioningField',
          },
          type: 'array',
        },
      ],
    },
    expected: { name: 'ConditioningField', cardinality: 'COLLECTION' },
  },
  {
    name: 'Nullable Collection ConditioningField',
    schema: {
      anyOf: [
        {
          items: {
            $ref: '#/components/schemas/ConditioningField',
          },
          type: 'array',
        },
        {
          type: 'null',
        },
      ],
    },
    expected: { name: 'ConditioningField', cardinality: 'COLLECTION' },
  },
  {
    name: 'SINGLE_OR_COLLECTION ConditioningField',
    schema: {
      anyOf: [
        {
          items: {
            $ref: '#/components/schemas/ConditioningField',
          },
          type: 'array',
        },
        {
          $ref: '#/components/schemas/ConditioningField',
        },
      ],
    },
    expected: { name: 'ConditioningField', cardinality: 'SINGLE_OR_COLLECTION' },
  },
  {
    name: 'Nullable SINGLE_OR_COLLECTION ConditioningField',
    schema: {
      anyOf: [
        {
          items: {
            $ref: '#/components/schemas/ConditioningField',
          },
          type: 'array',
        },
        {
          $ref: '#/components/schemas/ConditioningField',
        },
        {
          type: 'null',
        },
      ],
    },
    expected: { name: 'ConditioningField', cardinality: 'SINGLE_OR_COLLECTION' },
  },
];

const specialCases: ParseFieldTypeTestCase[] = [
  {
    name: 'String EnumField',
    schema: {
      type: 'string',
      enum: ['large', 'base', 'small'],
    },
    expected: { name: 'EnumField', cardinality: 'SINGLE' },
  },
  {
    name: 'String EnumField with one value',
    schema: {
      const: 'Some Value',
    },
    expected: { name: 'EnumField', cardinality: 'SINGLE' },
  },
  {
    name: 'Explicit ui_type (SchedulerField)',
    schema: {
      type: 'string',
      enum: ['ddim', 'ddpm', 'deis'],
      ui_type: 'SchedulerField',
    },
    expected: { name: 'EnumField', cardinality: 'SINGLE' },
  },
  {
    name: 'Explicit ui_type (AnyField)',
    schema: {
      type: 'string',
      enum: ['ddim', 'ddpm', 'deis'],
      ui_type: 'AnyField',
    },
    expected: { name: 'EnumField', cardinality: 'SINGLE' },
  },
  {
    name: 'Explicit ui_type (CollectionField)',
    schema: {
      type: 'string',
      enum: ['ddim', 'ddpm', 'deis'],
      ui_type: 'CollectionField',
    },
    expected: { name: 'EnumField', cardinality: 'SINGLE' },
  },
];

describe('refObjectToSchemaName', () => {
  it('parses ref object 1', () => {
    expect(
      refObjectToSchemaName({
        $ref: '#/components/schemas/ImageField',
      })
    ).toEqual('ImageField');
  });
  it('parses ref object 2', () => {
    expect(
      refObjectToSchemaName({
        $ref: '#/components/schemas/T2IAdapterModelField',
      })
    ).toEqual('T2IAdapterModelField');
  });
});

describe.concurrent('parseFieldType', () => {
  it.each(primitiveTypes)('parses primitive types ($name)', ({ schema, expected }: ParseFieldTypeTestCase) => {
    expect(parseFieldType(schema)).toEqual(expected);
  });
  it.each(complexTypes)('parses complex types ($name)', ({ schema, expected }: ParseFieldTypeTestCase) => {
    expect(parseFieldType(schema)).toEqual(expected);
  });
  it.each(specialCases)('parses special case types ($name)', ({ schema, expected }: ParseFieldTypeTestCase) => {
    expect(parseFieldType(schema)).toEqual(expected);
  });

  it('raises if it cannot extract a schema name from a ref', () => {
    expect(() =>
      parseFieldType({
        allOf: [
          {
            $ref: '#/components/schemas/',
          },
        ],
      })
    ).toThrowError(UnableToExtractSchemaNameFromRefError);
  });

  it('raises if it receives a union of mismatched types', () => {
    expect(() =>
      parseFieldType({
        anyOf: [
          {
            type: 'string',
          },
          {
            type: 'integer',
          },
        ],
      })
    ).toThrowError(UnsupportedUnionError);
  });

  it('raises if it receives a union of mismatched types (excluding null)', () => {
    expect(() =>
      parseFieldType({
        anyOf: [
          {
            type: 'string',
          },
          {
            type: 'integer',
          },
          {
            type: 'null',
          },
        ],
      })
    ).toThrowError(UnsupportedUnionError);
  });

  it('raises if it received an unsupported primitive type (object)', () => {
    expect(() =>
      parseFieldType({
        type: 'object',
      })
    ).toThrowError(UnsupportedPrimitiveTypeError);
  });

  it('raises if it received an unsupported primitive type (null)', () => {
    expect(() =>
      parseFieldType({
        type: 'null',
      })
    ).toThrowError(UnsupportedPrimitiveTypeError);
  });

  it('raises if it received an unsupported array item type (object)', () => {
    expect(() =>
      parseFieldType({
        items: {
          type: 'object',
        },
        type: 'array',
      })
    ).toThrowError(UnsupportedArrayItemType);
  });

  it('raises if it received an unsupported array item type (null)', () => {
    expect(() =>
      parseFieldType({
        items: {
          type: 'null',
        },
        type: 'array',
      })
    ).toThrowError(UnsupportedArrayItemType);
  });
});
