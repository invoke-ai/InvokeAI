import { SD1ModelState } from './models/sd1ModelSlice';
import { SD2ModelState } from './models/sd2ModelSlice';

/**
 * Models slice persist denylist
 */
export const modelsPersistDenylist:
  | (keyof SD1ModelState)[]
  | (keyof SD2ModelState)[] = ['entities', 'ids'];
