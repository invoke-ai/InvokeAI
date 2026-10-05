/**
 * Async generator resolution for batch nodes. Kept apart from `queries` and loaded on demand by the submit path and
 * the generator field, so neither the editor boot graph nor the workflow query surface carries it.
 */
export {
  getWorkflowGeneratorQueryOptions,
  resolveWorkflowGenerators,
  type WorkflowGeneratorQueryResult,
} from './data/generatorQueries';
