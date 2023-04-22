/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $Edge = {
  properties: {
    source: {
      type: 'all-of',
      description: `The connection for the edge's from node and field`,
      contains: [{
        type: 'EdgeConnection',
      }],
      isRequired: true,
    },
    destination: {
      type: 'all-of',
      description: `The connection for the edge's to node and field`,
      contains: [{
        type: 'EdgeConnection',
      }],
      isRequired: true,
    },
  },
} as const;
