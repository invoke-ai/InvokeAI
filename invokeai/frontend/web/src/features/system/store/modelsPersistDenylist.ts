import { SD1PipelineModelState } from './models/sd1PipelineModelSlice';
import { SD2PipelineModelState } from './models/sd2PipelineModelSlice';

/**
 * Models slice persist denylist
 */
export const modelsPersistDenylist:
  | (keyof SD1PipelineModelState)[]
  | (keyof SD2PipelineModelState)[] = ['entities', 'ids'];
