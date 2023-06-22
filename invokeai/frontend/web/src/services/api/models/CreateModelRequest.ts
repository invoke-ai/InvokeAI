/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { CkptModelInfo } from './CkptModelInfo';
import type { DiffusersModelInfo } from './DiffusersModelInfo';

export type CreateModelRequest = {
  /**
   * The name of the model
   */
  name: string;
  /**
   * The model info
   */
  info: (CkptModelInfo | DiffusersModelInfo);
};

