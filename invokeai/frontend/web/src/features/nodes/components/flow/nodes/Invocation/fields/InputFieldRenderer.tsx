import { Box, Text } from '@invoke-ai/ui-library';
import { useFieldInstance } from 'features/nodes/hooks/useFieldData';
import { useFieldTemplate } from 'features/nodes/hooks/useFieldTemplate';
import {
  isBoardFieldInputInstance,
  isBoardFieldInputTemplate,
  isBooleanFieldInputInstance,
  isBooleanFieldInputTemplate,
  isColorFieldInputInstance,
  isColorFieldInputTemplate,
  isControlNetModelFieldInputInstance,
  isControlNetModelFieldInputTemplate,
  isEnumFieldInputInstance,
  isEnumFieldInputTemplate,
  isFloatFieldInputInstance,
  isFloatFieldInputTemplate,
  isImageFieldInputInstance,
  isImageFieldInputTemplate,
  isIntegerFieldInputInstance,
  isIntegerFieldInputTemplate,
  isIPAdapterModelFieldInputInstance,
  isIPAdapterModelFieldInputTemplate,
  isLoRAModelFieldInputInstance,
  isLoRAModelFieldInputTemplate,
  isMainModelFieldInputInstance,
  isMainModelFieldInputTemplate,
  isSchedulerFieldInputInstance,
  isSchedulerFieldInputTemplate,
  isSDXLMainModelFieldInputInstance,
  isSDXLMainModelFieldInputTemplate,
  isSDXLRefinerModelFieldInputInstance,
  isSDXLRefinerModelFieldInputTemplate,
  isStringFieldInputInstance,
  isStringFieldInputTemplate,
  isT2IAdapterModelFieldInputInstance,
  isT2IAdapterModelFieldInputTemplate,
  isVAEModelFieldInputInstance,
  isVAEModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

import BoardFieldInputComponent from './inputs/BoardFieldInputComponent';
import BooleanFieldInputComponent from './inputs/BooleanFieldInputComponent';
import ColorFieldInputComponent from './inputs/ColorFieldInputComponent';
import ControlNetModelFieldInputComponent from './inputs/ControlNetModelFieldInputComponent';
import EnumFieldInputComponent from './inputs/EnumFieldInputComponent';
import ImageFieldInputComponent from './inputs/ImageFieldInputComponent';
import IPAdapterModelFieldInputComponent from './inputs/IPAdapterModelFieldInputComponent';
import LoRAModelFieldInputComponent from './inputs/LoRAModelFieldInputComponent';
import MainModelFieldInputComponent from './inputs/MainModelFieldInputComponent';
import NumberFieldInputComponent from './inputs/NumberFieldInputComponent';
import RefinerModelFieldInputComponent from './inputs/RefinerModelFieldInputComponent';
import SchedulerFieldInputComponent from './inputs/SchedulerFieldInputComponent';
import SDXLMainModelFieldInputComponent from './inputs/SDXLMainModelFieldInputComponent';
import StringFieldInputComponent from './inputs/StringFieldInputComponent';
import T2IAdapterModelFieldInputComponent from './inputs/T2IAdapterModelFieldInputComponent';
import VAEModelFieldInputComponent from './inputs/VAEModelFieldInputComponent';

type InputFieldProps = {
  nodeId: string;
  fieldName: string;
};

const InputFieldRenderer = ({ nodeId, fieldName }: InputFieldProps) => {
  const { t } = useTranslation();
  const fieldInstance = useFieldInstance(nodeId, fieldName);
  const fieldTemplate = useFieldTemplate(nodeId, fieldName, 'input');

  if (fieldTemplate?.fieldKind === 'output') {
    return (
      <Box p={2}>
        {t('nodes.outputFieldInInput')}: {fieldInstance?.type.name}
      </Box>
    );
  }

  if (isStringFieldInputInstance(fieldInstance) && isStringFieldInputTemplate(fieldTemplate)) {
    return <StringFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (isBooleanFieldInputInstance(fieldInstance) && isBooleanFieldInputTemplate(fieldTemplate)) {
    return <BooleanFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (
    (isIntegerFieldInputInstance(fieldInstance) && isIntegerFieldInputTemplate(fieldTemplate)) ||
    (isFloatFieldInputInstance(fieldInstance) && isFloatFieldInputTemplate(fieldTemplate))
  ) {
    return <NumberFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (isEnumFieldInputInstance(fieldInstance) && isEnumFieldInputTemplate(fieldTemplate)) {
    return <EnumFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (isImageFieldInputInstance(fieldInstance) && isImageFieldInputTemplate(fieldTemplate)) {
    return <ImageFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (isBoardFieldInputInstance(fieldInstance) && isBoardFieldInputTemplate(fieldTemplate)) {
    return <BoardFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (isMainModelFieldInputInstance(fieldInstance) && isMainModelFieldInputTemplate(fieldTemplate)) {
    return <MainModelFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (isSDXLRefinerModelFieldInputInstance(fieldInstance) && isSDXLRefinerModelFieldInputTemplate(fieldTemplate)) {
    return <RefinerModelFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (isVAEModelFieldInputInstance(fieldInstance) && isVAEModelFieldInputTemplate(fieldTemplate)) {
    return <VAEModelFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (isLoRAModelFieldInputInstance(fieldInstance) && isLoRAModelFieldInputTemplate(fieldTemplate)) {
    return <LoRAModelFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (isControlNetModelFieldInputInstance(fieldInstance) && isControlNetModelFieldInputTemplate(fieldTemplate)) {
    return <ControlNetModelFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (isIPAdapterModelFieldInputInstance(fieldInstance) && isIPAdapterModelFieldInputTemplate(fieldTemplate)) {
    return <IPAdapterModelFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (isT2IAdapterModelFieldInputInstance(fieldInstance) && isT2IAdapterModelFieldInputTemplate(fieldTemplate)) {
    return <T2IAdapterModelFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }
  if (isColorFieldInputInstance(fieldInstance) && isColorFieldInputTemplate(fieldTemplate)) {
    return <ColorFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (isSDXLMainModelFieldInputInstance(fieldInstance) && isSDXLMainModelFieldInputTemplate(fieldTemplate)) {
    return <SDXLMainModelFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (isSchedulerFieldInputInstance(fieldInstance) && isSchedulerFieldInputTemplate(fieldTemplate)) {
    return <SchedulerFieldInputComponent nodeId={nodeId} field={fieldInstance} fieldTemplate={fieldTemplate} />;
  }

  if (fieldInstance && fieldTemplate) {
    // Fallback for when there is no component for the type
    return null;
  }

  return (
    <Box p={1}>
      <Text fontSize="sm" fontWeight="semibold" color="error.300">
        {t('nodes.unknownFieldType', { type: fieldInstance?.type.name })}
      </Text>
    </Box>
  );
};

export default memo(InputFieldRenderer);
