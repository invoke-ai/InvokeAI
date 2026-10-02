import type { SystemStyleObject } from '@invoke-ai/ui-library';
import type { Opts as LinkifyOpts } from 'linkifyjs';

export const linkifySx: SystemStyleObject = {
  a: {
    fontWeight: 'semibold',
  },
  'a:hover': {
    textDecoration: 'underline',
  },
};

export const linkifyOptions: LinkifyOpts = {
  target: '_blank',
  rel: 'noopener noreferrer',
  validate: (value) => /^https?:\/\//.test(value),
};
