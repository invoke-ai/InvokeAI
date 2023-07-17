import { log } from 'app/logging/useLogger';
import { zControlNetModel } from 'features/parameters/types/parameterSchemas';
import { ControlNetModelField } from 'services/api/types';

const moduleLog = log.child({ module: 'models' });

export const modelIdToControlNetModelParam = (
  controlNetModelId: string
): ControlNetModelField | undefined => {
  const [base_model, model_type, model_name] = controlNetModelId.split('/');

  const result = zControlNetModel.safeParse({
    base_model,
    model_name,
  });

  if (!result.success) {
    moduleLog.error(
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
