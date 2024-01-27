import { logger } from 'app/logging/logger';
import { zParameterT2IAdapterModel } from 'features/parameters/types/parameterSchemas';
import type { T2IAdapterModelField } from 'services/api/types';

export const modelIdToT2IAdapterModelParam = (t2iAdapterModelId: string): T2IAdapterModelField | undefined => {
  const log = logger('models');
  const [base_model, _model_type, model_name] = t2iAdapterModelId.split('/');

  const result = zParameterT2IAdapterModel.safeParse({
    base_model,
    model_name,
  });

  if (!result.success) {
    log.error(
      {
        t2iAdapterModelId,
        errors: result.error.format(),
      },
      'Failed to parse T2I-Adapter model id'
    );

    return;
  }

  return result.data;
};
