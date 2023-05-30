/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ControlField = {
  properties: {
    image: {
      type: 'all-of',
      description: `processed image`,
      contains: [{
        type: 'ImageField',
      }],
      isRequired: true,
    },
    control_model: {
      type: 'string',
      description: `control model used`,
      isRequired: true,
    },
    control_weight: {
      type: 'number',
      description: `weight given to controlnet`,
      isRequired: true,
    },
    begin_step_percent: {
      type: 'number',
      description: `% of total steps at which controlnet is first applied`,
      isRequired: true,
      maximum: 1,
    },
    end_step_percent: {
      type: 'number',
      description: `% of total steps at which controlnet is last applied`,
      isRequired: true,
      maximum: 1,
    },
  },
} as const;
