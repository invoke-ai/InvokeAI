import { logger } from 'app/logging/logger';
import { zIPAdapterModel } from 'features/parameters/types/parameterSchemas';
import { IPAdapterModelField } from 'services/api/types';

export const modelIdToIPAdapterModelParam = (
  ipAdapterModelId: string
): IPAdapterModelField | undefined => {
  const log = logger('models');
  const [base_model, _model_type, model_name] = ipAdapterModelId.split('/');

  const result = zIPAdapterModel.safeParse({
    base_model,
    model_name,
  });

  if (!result.success) {
    log.error(
      {
        ipAdapterModelId,
        errors: result.error.format(),
      },
      'Failed to parse IP-Adapter model id'
    );

    return;
  }

  return result.data;
};
