/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ColorField = {
  properties: {
    'r': {
      type: 'number',
      description: `The red component`,
      isRequired: true,
      maximum: 255,
    },
    'b': {
      type: 'number',
      description: `The blue component`,
      isRequired: true,
      maximum: 255,
    },
    'g': {
      type: 'number',
      description: `The green component`,
      isRequired: true,
      maximum: 255,
    },
    'a': {
      type: 'number',
      description: `The alpha component`,
      maximum: 255,
    },
  },
} as const;
