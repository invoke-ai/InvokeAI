import { logger } from 'app/logging/logger';
import {
  SDXLRefinerModelParam,
  zSDXLRefinerModel,
} from 'features/parameters/types/parameterSchemas';

export const modelIdToSDXLRefinerModelParam = (
  mainModelId: string
): SDXLRefinerModelParam | undefined => {
  const log = logger('models');
  const [base_model, model_type, model_name] = mainModelId.split('/');

  const result = zSDXLRefinerModel.safeParse({
    base_model,
    model_name,
    model_type,
  });

  if (!result.success) {
    log.error(
      {
        mainModelId,
        errors: result.error.format(),
      },
      'Failed to parse main model id'
    );
    return;
  }

  return result.data;
};
