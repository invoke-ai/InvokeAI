import { isEqual } from 'lodash-es';

export const defaultSelectorOptions = {
  memoizeOptions: {
    resultEqualityCheck: isEqual,
  },
};
