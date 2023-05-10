/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { VaeRepo } from './VaeRepo';

export type DiffusersModelInfo = {
  /**
   * A description of the model
   */
  description?: string;
  format?: 'diffusers';
  /**
   * The VAE repo to use for this model
   */
  vae?: VaeRepo;
  /**
   * The repo ID to use for this model
   */
  repo_id?: string;
  /**
   * The path to the model
   */
  path?: string;
};

