/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $LatentsField = {
  description: `A latents field used for passing latents between invocations`,
  properties: {
    latents_name: {
      type: 'string',
      description: `The name of the latents`,
      isRequired: true,
    },
  },
} as const;
