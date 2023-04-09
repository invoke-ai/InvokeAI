import { map, reduce } from 'lodash';
import { NodeTypes } from 'reactflow';
import { FieldConfig } from './types';
import { buildInvocations } from './util/buildInvocations';

// here we fetch the schema, parse it and output all the constants

export const PRIMITIVE_FIELDS = [
  'integer',
  'number',
  'boolean',
  'string',
  'object',
  'array',
];

export const AVAILABLE_COLORS = [
  'red',
  'orange',
  'yellow',
  'green',
  'blue',
  'purple',
  'pink',
  'teal',
  'cyan',
];

// export const { invocations: INVOCATIONS, fieldTypes } =
//   await buildInvocations();

// export const INVOCATION_NAMES: (keyof typeof INVOCATIONS)[] = map(
//   INVOCATIONS,
//   (_, key) => key
// );

// export const NODE_TYPES = reduce(
//   INVOCATIONS,
//   (acc, val, key) => {
//     acc[val.title] = val.component;
//     return acc;
//   },
//   {} as NodeTypes
// );

// export const NODE_TYPE_NAMES: (keyof typeof NODE_TYPES)[] = map(
//   NODE_TYPES,
//   (_, key) => key
// );

// // all field types, maybe we can dynamically generate this in the future?
// export const FIELDS = fieldTypes.reduce<FieldConfig>((acc, val, i) => {
//   let color = AVAILABLE_COLORS[i];
//   if (!color) {
//     color = 'gray';
//   }

//   acc[val] = {
//     color,
//     isPrimitive: PRIMITIVE_FIELDS.includes(val),
//   };

//   return acc;
// }, {});

// // helper array of all field names
// export const FIELD_NAMES: (keyof typeof FIELDS)[] = map(
//   FIELDS,
//   (_, key) => key
// );

// console.log('INVOCATIONS', INVOCATIONS);
// console.log('INVOCATION_NAMES', INVOCATION_NAMES);
// console.log('FIELDS', FIELDS);
// console.log('FIELD_NAMES', FIELD_NAMES);
// console.log('NODE_TYPES', NODE_TYPES);
// console.log('NODE_TYPE_NAMES', NODE_TYPE_NAMES);
