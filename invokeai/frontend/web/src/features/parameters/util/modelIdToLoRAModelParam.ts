import { LoRAModelParam, zLoRAModel } from '../types/parameterSchemas';
import { log } from 'app/logging/useLogger';

const moduleLog = log.child({ module: 'models' });

export const modelIdToLoRAModelParam = (
  loraModelId: string
): LoRAModelParam | undefined => {
  const [base_model, model_type, model_name] = loraModelId.split('/');

  const result = zLoRAModel.safeParse({
    base_model,
    model_name,
  });

  if (!result.success) {
    moduleLog.error(
      {
        loraModelId,
        errors: result.error.format(),
      },
      'Failed to parse LoRA model id'
    );
    return;
  }

  return result.data;
};
