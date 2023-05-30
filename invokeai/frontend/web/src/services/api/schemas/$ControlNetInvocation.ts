/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $ControlNetInvocation = {
  description: `Collects ControlNet info to pass to other nodes`,
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
      description: `image to process`,
      contains: [{
        type: 'ImageField',
      }],
    },
    control_model: {
      type: 'Enum',
    },
    control_weight: {
      type: 'number',
      description: `weight given to controlnet`,
      maximum: 1,
    },
    begin_step_percent: {
      type: 'number',
      description: `% of total steps at which controlnet is first applied`,
      maximum: 1,
    },
    end_step_percent: {
      type: 'number',
      description: `% of total steps at which controlnet is last applied`,
      maximum: 1,
    },
  },
} as const;
