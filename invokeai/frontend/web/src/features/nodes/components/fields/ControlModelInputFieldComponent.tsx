import { Select } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  ControlModelInputFieldTemplate,
  ControlModelInputFieldValue,
} from 'features/nodes/types/types';
import { ChangeEvent, ReactNode, memo, useEffect, useState } from 'react';
import { ModelsService } from 'services/api';
import { FieldComponentProps } from './types';

const ControlModelInputFieldComponent = (
  props: FieldComponentProps<
    ControlModelInputFieldValue,
    ControlModelInputFieldTemplate
  >
) => {
  const { nodeId, field } = props;

  const dispatch = useAppDispatch();

  const [controlnetModels, setControlNetModels] = useState<string[] | null>(
    null
  );

  const getControlNetModels = async () => {
    const response = await ModelsService.listControlnetModels();
    setControlNetModels(Object.keys(response['controlnet_models']));
  };

  useEffect(() => {
    getControlNetModels();
  }, []);

  const handleValueChanged = (e: ChangeEvent<HTMLSelectElement>) => {
    dispatch(
      fieldValueChanged({
        nodeId,
        fieldName: field.name,
        value: { control_model: e.target.value },
      })
    );
  };

  const renderControlNetModels = () => {
    const controlnetModelsToRender: ReactNode[] = [];
    if (controlnetModels) {
      Object.keys(controlnetModels).forEach((index) => {
        const controlnetModel = controlnetModels[Number(index)];
        controlnetModelsToRender.push(
          <option key={controlnetModel} value={controlnetModel}>
            {controlnetModel}
          </option>
        );
      });
    }
    return controlnetModelsToRender;
  };
  return (
    <Select onChange={handleValueChanged} value={field.value?.control_model}>
      {renderControlNetModels()}
    </Select>
  );
};

export default memo(ControlModelInputFieldComponent);
