/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { ModelError } from './ModelError';

export type TextualInversionModelConfig = {
  path: string;
  description?: string;
  format: null;
  default?: boolean;
  error?: ModelError;
};
