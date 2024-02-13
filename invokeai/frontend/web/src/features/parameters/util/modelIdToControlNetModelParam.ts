import { logger } from 'app/logging/logger';
import { zParameterControlNetModel } from 'features/parameters/types/parameterSchemas';
import type { ControlNetModelField } from 'services/api/types';

export const modelIdToControlNetModelParam = (controlNetModelId: string): ControlNetModelField | undefined => {
  const log = logger('models');
  const [base_model, _model_type, model_name] = controlNetModelId.split('/');

  const result = zParameterControlNetModel.safeParse({
    base_model,
    model_name,
  });

  if (!result.success) {
    log.error(
      {
        controlNetModelId,
        errors: result.error.format(),
      },
      'Failed to parse ControlNet model id'
    );

    return;
  }

  return result.data;
};
