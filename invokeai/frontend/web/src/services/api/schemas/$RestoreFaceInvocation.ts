/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $RestoreFaceInvocation = {
  description: `Restores faces in an image.`,
  properties: {
    id: {
      type: 'string',
      description: `The id of this node. Must be unique among all nodes.`,
      isRequired: true,
    },
    type: {
      type: 'Enum',
    },
    image: {
      type: 'all-of',
      description: `The input image`,
      contains: [{
        type: 'ImageField',
      }],
    },
    strength: {
      type: 'number',
      description: `The strength of the restoration`,
      maximum: 1,
    },
  },
} as const;
