/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { CkptModelInfo } from './CkptModelInfo';
import type { DiffusersModelInfo } from './DiffusersModelInfo';

export type ModelsList = {
  models: Record<string, (CkptModelInfo | DiffusersModelInfo)>;
};

