import { skipToken } from '@reduxjs/toolkit/query';
import { useGetModelConfigQuery } from 'services/api/endpoints/models';
import type { AnyModelConfig } from 'services/api/types';

export const useGetModelConfigWithTypeGuard = <T extends AnyModelConfig>(
  key: string | typeof skipToken,
  typeGuard: (config: AnyModelConfig) => config is T
) => {
  const result = useGetModelConfigQuery(key ?? skipToken, {
    selectFromResult: (result) => {
      const modelConfig = result.currentData;
      return {
        ...result,
        modelConfig: modelConfig && typeGuard(modelConfig) ? modelConfig : undefined,
      };
    },
  });

  return result;
};
