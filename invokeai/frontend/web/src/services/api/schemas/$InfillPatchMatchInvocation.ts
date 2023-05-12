/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $InfillPatchMatchInvocation = {
  description: `Infills transparent areas of an image using the PatchMatch algorithm`,
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
      description: `The image to infill`,
      contains: [{
        type: 'ImageField',
      }],
    },
  },
} as const;
