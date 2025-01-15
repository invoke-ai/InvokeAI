import { ImageFieldCollectionInputComponent } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ImageFieldCollectionInputComponent';
import ModelIdentifierFieldInputComponent from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/ModelIdentifierFieldInputComponent';
import { NumberFieldCollectionInputComponent } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/NumberFieldCollectionInputComponent';
import { StringFieldCollectionInputComponent } from 'features/nodes/components/flow/nodes/Invocation/fields/inputs/StringFieldCollectionInputComponent';
import { useFieldInputInstance } from 'features/nodes/hooks/useFieldInputInstance';
import { useFieldInputTemplate } from 'features/nodes/hooks/useFieldInputTemplate';
import {
  isBoardFieldInputInstance,
  isBoardFieldInputTemplate,
  isBooleanFieldInputInstance,
  isBooleanFieldInputTemplate,
  isCLIPEmbedModelFieldInputInstance,
  isCLIPEmbedModelFieldInputTemplate,
  isCLIPGEmbedModelFieldInputInstance,
  isCLIPGEmbedModelFieldInputTemplate,
  isCLIPLEmbedModelFieldInputInstance,
  isCLIPLEmbedModelFieldInputTemplate,
  isColorFieldInputInstance,
  isColorFieldInputTemplate,
  isControlLoRAModelFieldInputInstance,
  isControlLoRAModelFieldInputTemplate,
  isControlNetModelFieldInputInstance,
  isControlNetModelFieldInputTemplate,
  isEnumFieldInputInstance,
  isEnumFieldInputTemplate,
  isFloatFieldCollectionInputInstance,
  isFloatFieldCollectionInputTemplate,
  isFloatFieldInputInstance,
  isFloatFieldInputTemplate,
  isFluxMainModelFieldInputInstance,
  isFluxMainModelFieldInputTemplate,
  isFluxVAEModelFieldInputInstance,
  isFluxVAEModelFieldInputTemplate,
  isImageFieldCollectionInputInstance,
  isImageFieldCollectionInputTemplate,
  isImageFieldInputInstance,
  isImageFieldInputTemplate,
  isIntegerFieldCollectionInputInstance,
  isIntegerFieldCollectionInputTemplate,
  isIntegerFieldInputInstance,
  isIntegerFieldInputTemplate,
  isIPAdapterModelFieldInputInstance,
  isIPAdapterModelFieldInputTemplate,
  isLoRAModelFieldInputInstance,
  isLoRAModelFieldInputTemplate,
  isMainModelFieldInputInstance,
  isMainModelFieldInputTemplate,
  isModelIdentifierFieldInputInstance,
  isModelIdentifierFieldInputTemplate,
  isSchedulerFieldInputInstance,
  isSchedulerFieldInputTemplate,
  isSD3MainModelFieldInputInstance,
  isSD3MainModelFieldInputTemplate,
  isSDXLMainModelFieldInputInstance,
  isSDXLMainModelFieldInputTemplate,
  isSDXLRefinerModelFieldInputInstance,
  isSDXLRefinerModelFieldInputTemplate,
  isSpandrelImageToImageModelFieldInputInstance,
  isSpandrelImageToImageModelFieldInputTemplate,
  isStringFieldCollectionInputInstance,
  isStringFieldCollectionInputTemplate,
  isStringFieldInputInstance,
  isStringFieldInputTemplate,
  isT2IAdapterModelFieldInputInstance,
  isT2IAdapterModelFieldInputTemplate,
  isT5EncoderModelFieldInputInstance,
  isT5EncoderModelFieldInputTemplate,
  isVAEModelFieldInputInstance,
  isVAEModelFieldInputTemplate,
} from 'features/nodes/types/field';
import { memo } from 'react';

import BoardFieldInputComponent from './inputs/BoardFieldInputComponent';
import BooleanFieldInputComponent from './inputs/BooleanFieldInputComponent';
import CLIPEmbedModelFieldInputComponent from './inputs/CLIPEmbedModelFieldInputComponent';
import CLIPGEmbedModelFieldInputComponent from './inputs/CLIPGEmbedModelFieldInputComponent';
import CLIPLEmbedModelFieldInputComponent from './inputs/CLIPLEmbedModelFieldInputComponent';
import ColorFieldInputComponent from './inputs/ColorFieldInputComponent';
import ControlLoRAModelFieldInputComponent from './inputs/ControlLoraModelFieldInputComponent';
import ControlNetModelFieldInputComponent from './inputs/ControlNetModelFieldInputComponent';
import EnumFieldInputComponent from './inputs/EnumFieldInputComponent';
import FluxMainModelFieldInputComponent from './inputs/FluxMainModelFieldInputComponent';
import FluxVAEModelFieldInputComponent from './inputs/FluxVAEModelFieldInputComponent';
import ImageFieldInputComponent from './inputs/ImageFieldInputComponent';
import IPAdapterModelFieldInputComponent from './inputs/IPAdapterModelFieldInputComponent';
import LoRAModelFieldInputComponent from './inputs/LoRAModelFieldInputComponent';
import MainModelFieldInputComponent from './inputs/MainModelFieldInputComponent';
import NumberFieldInputComponent from './inputs/NumberFieldInputComponent';
import RefinerModelFieldInputComponent from './inputs/RefinerModelFieldInputComponent';
import SchedulerFieldInputComponent from './inputs/SchedulerFieldInputComponent';
import SD3MainModelFieldInputComponent from './inputs/SD3MainModelFieldInputComponent';
import SDXLMainModelFieldInputComponent from './inputs/SDXLMainModelFieldInputComponent';
import SpandrelImageToImageModelFieldInputComponent from './inputs/SpandrelImageToImageModelFieldInputComponent';
import StringFieldInputComponent from './inputs/StringFieldInputComponent';
import T2IAdapterModelFieldInputComponent from './inputs/T2IAdapterModelFieldInputComponent';
import T5EncoderModelFieldInputComponent from './inputs/T5EncoderModelFieldInputComponent';
import VAEModelFieldInputComponent from './inputs/VAEModelFieldInputComponent';

type InputFieldProps = {
  nodeId: string;
  fieldName: string;
  isLinearView: boolean;
};

const InputFieldRenderer = ({ nodeId, fieldName, isLinearView }: InputFieldProps) => {
  const fieldInstance = useFieldInputInstance(nodeId, fieldName);
  const fieldTemplate = useFieldInputTemplate(nodeId, fieldName);

  if (isStringFieldCollectionInputInstance(fieldInstance) && isStringFieldCollectionInputTemplate(fieldTemplate)) {
    return (
      <StringFieldCollectionInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isStringFieldInputInstance(fieldInstance) && isStringFieldInputTemplate(fieldTemplate)) {
    return (
      <StringFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isBooleanFieldInputInstance(fieldInstance) && isBooleanFieldInputTemplate(fieldTemplate)) {
    return (
      <BooleanFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isIntegerFieldInputInstance(fieldInstance) && isIntegerFieldInputTemplate(fieldTemplate)) {
    return (
      <NumberFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isFloatFieldInputInstance(fieldInstance) && isFloatFieldInputTemplate(fieldTemplate)) {
    return (
      <NumberFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isIntegerFieldCollectionInputInstance(fieldInstance) && isIntegerFieldCollectionInputTemplate(fieldTemplate)) {
    return (
      <NumberFieldCollectionInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isFloatFieldCollectionInputInstance(fieldInstance) && isFloatFieldCollectionInputTemplate(fieldTemplate)) {
    return (
      <NumberFieldCollectionInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isEnumFieldInputInstance(fieldInstance) && isEnumFieldInputTemplate(fieldTemplate)) {
    return (
      <EnumFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isImageFieldCollectionInputInstance(fieldInstance) && isImageFieldCollectionInputTemplate(fieldTemplate)) {
    return (
      <ImageFieldCollectionInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isImageFieldInputInstance(fieldInstance) && isImageFieldInputTemplate(fieldTemplate)) {
    return (
      <ImageFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isBoardFieldInputInstance(fieldInstance) && isBoardFieldInputTemplate(fieldTemplate)) {
    return (
      <BoardFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isMainModelFieldInputInstance(fieldInstance) && isMainModelFieldInputTemplate(fieldTemplate)) {
    return (
      <MainModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isModelIdentifierFieldInputInstance(fieldInstance) && isModelIdentifierFieldInputTemplate(fieldTemplate)) {
    return (
      <ModelIdentifierFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isSDXLRefinerModelFieldInputInstance(fieldInstance) && isSDXLRefinerModelFieldInputTemplate(fieldTemplate)) {
    return (
      <RefinerModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isVAEModelFieldInputInstance(fieldInstance) && isVAEModelFieldInputTemplate(fieldTemplate)) {
    return (
      <VAEModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isT5EncoderModelFieldInputInstance(fieldInstance) && isT5EncoderModelFieldInputTemplate(fieldTemplate)) {
    return (
      <T5EncoderModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }
  if (isCLIPEmbedModelFieldInputInstance(fieldInstance) && isCLIPEmbedModelFieldInputTemplate(fieldTemplate)) {
    return (
      <CLIPEmbedModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isCLIPLEmbedModelFieldInputInstance(fieldInstance) && isCLIPLEmbedModelFieldInputTemplate(fieldTemplate)) {
    return (
      <CLIPLEmbedModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isCLIPGEmbedModelFieldInputInstance(fieldInstance) && isCLIPGEmbedModelFieldInputTemplate(fieldTemplate)) {
    return (
      <CLIPGEmbedModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isControlLoRAModelFieldInputInstance(fieldInstance) && isControlLoRAModelFieldInputTemplate(fieldTemplate)) {
    return (
      <ControlLoRAModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isFluxVAEModelFieldInputInstance(fieldInstance) && isFluxVAEModelFieldInputTemplate(fieldTemplate)) {
    return (
      <FluxVAEModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isLoRAModelFieldInputInstance(fieldInstance) && isLoRAModelFieldInputTemplate(fieldTemplate)) {
    return (
      <LoRAModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isControlNetModelFieldInputInstance(fieldInstance) && isControlNetModelFieldInputTemplate(fieldTemplate)) {
    return (
      <ControlNetModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isIPAdapterModelFieldInputInstance(fieldInstance) && isIPAdapterModelFieldInputTemplate(fieldTemplate)) {
    return (
      <IPAdapterModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isT2IAdapterModelFieldInputInstance(fieldInstance) && isT2IAdapterModelFieldInputTemplate(fieldTemplate)) {
    return (
      <T2IAdapterModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (
    isSpandrelImageToImageModelFieldInputInstance(fieldInstance) &&
    isSpandrelImageToImageModelFieldInputTemplate(fieldTemplate)
  ) {
    return (
      <SpandrelImageToImageModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isColorFieldInputInstance(fieldInstance) && isColorFieldInputTemplate(fieldTemplate)) {
    return (
      <ColorFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isFluxMainModelFieldInputInstance(fieldInstance) && isFluxMainModelFieldInputTemplate(fieldTemplate)) {
    return (
      <FluxMainModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isSD3MainModelFieldInputInstance(fieldInstance) && isSD3MainModelFieldInputTemplate(fieldTemplate)) {
    return (
      <SD3MainModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isSDXLMainModelFieldInputInstance(fieldInstance) && isSDXLMainModelFieldInputTemplate(fieldTemplate)) {
    return (
      <SDXLMainModelFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (isSchedulerFieldInputInstance(fieldInstance) && isSchedulerFieldInputTemplate(fieldTemplate)) {
    return (
      <SchedulerFieldInputComponent
        nodeId={nodeId}
        field={fieldInstance}
        fieldTemplate={fieldTemplate}
        isLinearView={isLinearView}
      />
    );
  }

  if (fieldTemplate) {
    // Fallback for when there is no component for the type
    return null;
  }
};

export default memo(InputFieldRenderer);
