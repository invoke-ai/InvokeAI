/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $EdgeConnection = {
  properties: {
    node_id: {
      type: 'string',
      description: `The id of the node for this edge connection`,
      isRequired: true,
    },
    field: {
      type: 'string',
      description: `The field for this connection`,
      isRequired: true,
    },
  },
} as const;
