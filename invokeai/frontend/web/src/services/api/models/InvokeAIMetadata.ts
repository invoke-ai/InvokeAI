/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */

import type { MetadataImageField } from './MetadataImageField';
import type { MetadataLatentsField } from './MetadataLatentsField';

export type InvokeAIMetadata = {
  session_id?: string;
  node?: Record<string, (string | number | boolean | MetadataImageField | MetadataLatentsField)>;
};

