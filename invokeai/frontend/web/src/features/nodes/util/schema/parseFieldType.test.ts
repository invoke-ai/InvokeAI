import {
  UnableToExtractSchemaNameFromRefError,
  UnsupportedArrayItemType,
  UnsupportedPrimitiveTypeError,
  UnsupportedUnionError,
} from 'features/nodes/types/error';
import type { InvocationFieldSchema, OpenAPIV3_1SchemaOrRef } from 'features/nodes/types/openapi';
import { parseFieldType, refObjectToSchemaName } from 'features/nodes/util/schema/parseFieldType';
import { describe, expect, it } from 'vitest';

type ParseFieldTypeTestCase = {
  name: string;
  schema: OpenAPIV3_1SchemaOrRef | InvocationFieldSchema;
  expected: { name: string; isCollection: boolean; isCollectionOrScalar: boolean };
};

const primitiveTypes: ParseFieldTypeTestCase[] = [
  {
    name: 'Scalar IntegerField',
    schema: { type: 'integer' },
    expected: { name: 'IntegerField', isCollection: false, isCollectionOrScalar: false },
  },
  {
    name: 'Scalar FloatField',
    schema: { type: 'number' },
    expected: { name: 'FloatField', isCollection: false, isCollectionOrScalar: false },
  },
  {
    name: 'Scalar StringField',
    schema: { type: 'string' },
    expected: { name: 'StringField', isCollection: false, isCollectionOrScalar: false },
  },
  {
    name: 'Scalar BooleanField',
    schema: { type: 'boolean' },
    expected: { name: 'BooleanField', isCollection: false, isCollectionOrScalar: false },
  },
  {
    name: 'Collection IntegerField',
    schema: { items: { type: 'integer' }, type: 'array' },
    expected: { name: 'IntegerField', isCollection: true, isCollectionOrScalar: false },
  },
  {
    name: 'Collection FloatField',
    schema: { items: { type: 'number' }, type: 'array' },
    expected: { name: 'FloatField', isCollection: true, isCollectionOrScalar: false },
  },
  {
    name: 'Collection StringField',
    schema: { items: { type: 'string' }, type: 'array' },
    expected: { name: 'StringField', isCollection: true, isCollectionOrScalar: false },
  },
  {
    name: 'Collection BooleanField',
    schema: { items: { type: 'boolean' }, type: 'array' },
    expected: { name: 'BooleanField', isCollection: true, isCollectionOrScalar: false },
  },
  {
    name: 'CollectionOrScalar IntegerField',
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
    expected: { name: 'IntegerField', isCollection: false, isCollectionOrScalar: true },
  },
  {
    name: 'CollectionOrScalar FloatField',
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
    expected: { name: 'FloatField', isCollection: false, isCollectionOrScalar: true },
  },
  {
    name: 'CollectionOrScalar StringField',
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
    expected: { name: 'StringField', isCollection: false, isCollectionOrScalar: true },
  },
  {
    name: 'CollectionOrScalar BooleanField',
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
    expected: { name: 'BooleanField', isCollection: false, isCollectionOrScalar: true },
  },
];

const complexTypes: ParseFieldTypeTestCase[] = [
  {
    name: 'Scalar ConditioningField',
    schema: {
      allOf: [
        {
          $ref: '#/components/schemas/ConditioningField',
        },
      ],
    },
    expected: { name: 'ConditioningField', isCollection: false, isCollectionOrScalar: false },
  },
  {
    name: 'Nullable Scalar ConditioningField',
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
    expected: { name: 'ConditioningField', isCollection: false, isCollectionOrScalar: false },
  },
  {
    name: 'Collection ConditioningField',
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
    expected: { name: 'ConditioningField', isCollection: true, isCollectionOrScalar: false },
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
    expected: { name: 'ConditioningField', isCollection: true, isCollectionOrScalar: false },
  },
  {
    name: 'CollectionOrScalar ConditioningField',
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
    expected: { name: 'ConditioningField', isCollection: false, isCollectionOrScalar: true },
  },
  {
    name: 'Nullable CollectionOrScalar ConditioningField',
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
    expected: { name: 'ConditioningField', isCollection: false, isCollectionOrScalar: true },
  },
];

const specialCases: ParseFieldTypeTestCase[] = [
  {
    name: 'String EnumField',
    schema: {
      type: 'string',
      enum: ['large', 'base', 'small'],
    },
    expected: { name: 'EnumField', isCollection: false, isCollectionOrScalar: false },
  },
  {
    name: 'String EnumField with one value',
    schema: {
      const: 'Some Value',
    },
    expected: { name: 'EnumField', isCollection: false, isCollectionOrScalar: false },
  },
  {
    name: 'Explicit ui_type (SchedulerField)',
    schema: {
      type: 'string',
      enum: ['ddim', 'ddpm', 'deis'],
      ui_type: 'SchedulerField',
    },
    expected: { name: 'SchedulerField', isCollection: false, isCollectionOrScalar: false },
  },
  {
    name: 'Explicit ui_type (AnyField)',
    schema: {
      type: 'string',
      enum: ['ddim', 'ddpm', 'deis'],
      ui_type: 'AnyField',
    },
    expected: { name: 'AnyField', isCollection: false, isCollectionOrScalar: false },
  },
  {
    name: 'Explicit ui_type (CollectionField)',
    schema: {
      type: 'string',
      enum: ['ddim', 'ddpm', 'deis'],
      ui_type: 'CollectionField',
    },
    expected: { name: 'CollectionField', isCollection: true, isCollectionOrScalar: false },
  },
];

describe('refObjectToSchemaName', async () => {
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

describe.concurrent('parseFieldType', async () => {
  it.each(primitiveTypes)('parses primitive types ($name)', ({ schema, expected }) => {
    expect(parseFieldType(schema)).toEqual(expected);
  });
  it.each(complexTypes)('parses complex types ($name)', ({ schema, expected }) => {
    expect(parseFieldType(schema)).toEqual(expected);
  });
  it.each(specialCases)('parses special case types ($name)', ({ schema, expected }) => {
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
