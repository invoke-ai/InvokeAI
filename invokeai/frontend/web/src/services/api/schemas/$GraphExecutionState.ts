/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $GraphExecutionState = {
  description: `Tracks the state of a graph execution`,
  properties: {
    id: {
      type: 'string',
      description: `The id of the execution state`,
      isRequired: true,
    },
    graph: {
      type: 'all-of',
      description: `The graph being executed`,
      contains: [{
        type: 'Graph',
      }],
      isRequired: true,
    },
    execution_graph: {
      type: 'all-of',
      description: `The expanded graph of activated and executed nodes`,
      contains: [{
        type: 'Graph',
      }],
      isRequired: true,
    },
    executed: {
      type: 'array',
      contains: {
        type: 'string',
      },
      isRequired: true,
    },
    executed_history: {
      type: 'array',
      contains: {
        type: 'string',
      },
      isRequired: true,
    },
    results: {
      type: 'dictionary',
      contains: {
        type: 'one-of',
        contains: [{
          type: 'ImageOutput',
        }, {
          type: 'MaskOutput',
        }, {
          type: 'CompelOutput',
        }, {
          type: 'LatentsOutput',
        }, {
          type: 'NoiseOutput',
        }, {
          type: 'IntOutput',
        }, {
          type: 'PromptOutput',
        }, {
          type: 'IntCollectionOutput',
        }, {
          type: 'GraphInvocationOutput',
        }, {
          type: 'IterateInvocationOutput',
        }, {
          type: 'CollectInvocationOutput',
        }],
      },
      isRequired: true,
    },
    errors: {
      type: 'dictionary',
      contains: {
        type: 'string',
      },
      isRequired: true,
    },
    prepared_source_mapping: {
      type: 'dictionary',
      contains: {
        type: 'string',
      },
      isRequired: true,
    },
    source_prepared_mapping: {
      type: 'dictionary',
      contains: {
        type: 'array',
        contains: {
          type: 'string',
        },
      },
      isRequired: true,
    },
  },
} as const;
