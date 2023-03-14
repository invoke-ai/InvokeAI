/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
export const $GraphExecutionState = {
  description: `Tracks the state of a graph execution`,
  properties: {
    id: {
      type: 'string',
      description: `The id of the execution state`,
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
    },
    executed: {
      type: 'array',
      contains: {
        type: 'string',
      },
    },
    executed_history: {
      type: 'array',
      contains: {
        type: 'string',
      },
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
          type: 'PromptOutput',
        }, {
          type: 'GraphInvocationOutput',
        }, {
          type: 'IterateInvocationOutput',
        }, {
          type: 'CollectInvocationOutput',
        }],
      },
    },
    errors: {
      type: 'dictionary',
      contains: {
        type: 'string',
      },
    },
    prepared_source_mapping: {
      type: 'dictionary',
      contains: {
        type: 'string',
      },
    },
    source_prepared_mapping: {
      type: 'dictionary',
      contains: {
        type: 'array',
        contains: {
          type: 'string',
        },
      },
    },
  },
} as const;
