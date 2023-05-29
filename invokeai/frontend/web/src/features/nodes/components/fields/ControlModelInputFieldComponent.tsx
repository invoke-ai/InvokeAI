import { Flex, Select } from '@chakra-ui/react';
import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { fieldValueChanged } from 'features/nodes/store/nodesSlice';
import {
  ControlModelInputFieldTemplate,
  ControlModelInputFieldValue,
} from 'features/nodes/types/types';
import { ChangeEvent, ReactNode, memo, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { BiRefresh } from 'react-icons/bi';
import { ModelsService } from 'services/api';
import { FieldComponentProps } from './types';

const ControlModelInputFieldComponent = (
  props: FieldComponentProps<
    ControlModelInputFieldValue,
    ControlModelInputFieldTemplate
  >
) => {
  const { nodeId, field } = props;

  const { t } = useTranslation();

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
      controlnetModels.forEach((controlnetModel) => {
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
    <Flex alignItems="center" w="full">
      <Select onChange={handleValueChanged} value={field.value?.control_model}>
        {renderControlNetModels()}
      </Select>
      <IAIIconButton
        aria-label={t('common.refresh')}
        icon={<BiRefresh size={18} />}
        size="sm"
        onClick={() => getControlNetModels()}
        sx={{
          background: 'none',
          _hover: {
            background: 'none',
          },
        }}
      />
    </Flex>
  );
};

export default memo(ControlModelInputFieldComponent);
