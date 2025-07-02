import {
  Alert,
  AlertDescription,
  AlertIcon,
  Box,
  Flex,
  FormControl,
  FormHelperText,
  FormLabel,
  Input,
  Select,
  Text,
  Textarea,
} from '@invoke-ai/ui-library';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiWarningBold } from 'react-icons/pi';

type ManualModelConfig = {
  name?: string;
  description?: string;
  type?: string;
  base?: string;
  format?: string;
  prediction_type?: string;
  variant?: string;
};

type ManualModelConfigPanelProps = {
  config: ManualModelConfig;
  onChange: (config: ManualModelConfig) => void;
};

export const ManualModelConfigPanel = memo(({ config, onChange }: ManualModelConfigPanelProps) => {
  const { t } = useTranslation();

  const updateConfig = useCallback(
    (key: keyof ManualModelConfig, value: string) => {
      onChange({
        ...config,
        [key]: value || undefined,
      });
    },
    [config, onChange]
  );

  const onNameChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => updateConfig('name', e.target.value),
    [updateConfig]
  );

  const onDescriptionChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => updateConfig('description', e.target.value),
    [updateConfig]
  );

  const onTypeChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => updateConfig('type', e.target.value),
    [updateConfig]
  );

  const onBaseChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => updateConfig('base', e.target.value),
    [updateConfig]
  );

  const onFormatChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => updateConfig('format', e.target.value),
    [updateConfig]
  );

  const onPredictionTypeChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => updateConfig('prediction_type', e.target.value),
    [updateConfig]
  );

  const onVariantChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => updateConfig('variant', e.target.value),
    [updateConfig]
  );

  return (
    <Box layerStyle="second" borderRadius="base" p={4}>
      <Flex flexDir="column" gap={4}>
        <Alert status="warning" borderRadius="base">
          <AlertIcon as={PiWarningBold} />
          <AlertDescription>
            <Flex flexDir="column" gap={2}>
              <Text fontWeight="semibold">{t('modelManager.manualConfigWarningTitle')}</Text>
              <Text fontSize="sm">{t('modelManager.manualConfigWarningDesc')}</Text>
              <Box>
                <Text fontSize="sm" fontWeight="medium">
                  {t('modelManager.manualConfigRisks')}:
                </Text>
                <Text fontSize="sm" as="ul" pl={4}>
                  <Text as="li">• {t('modelManager.riskNoProbing')}</Text>
                  <Text as="li">• {t('modelManager.riskMayNotWork')}</Text>
                  <Text as="li">• {t('modelManager.riskIncorrectConfig')}</Text>
                </Text>
              </Box>
            </Flex>
          </AlertDescription>
        </Alert>

        <Flex flexDir="column" gap={4}>
          <FormControl>
            <FormLabel>{t('modelManager.modelName')}</FormLabel>
            <Input
              value={config.name || ''}
              onChange={onNameChange}
              placeholder={t('modelManager.modelNamePlaceholder')}
            />
            <FormHelperText>{t('modelManager.modelNameHelper')}</FormHelperText>
          </FormControl>

          <FormControl>
            <FormLabel>{t('modelManager.description')}</FormLabel>
            <Textarea
              value={config.description || ''}
              onChange={onDescriptionChange}
              placeholder={t('modelManager.descriptionPlaceholder')}
              rows={2}
            />
          </FormControl>

          <Flex gap={4}>
            <FormControl flex={1}>
              <FormLabel>{t('modelManager.modelType')}</FormLabel>
              <Select value={config.type || ''} onChange={onTypeChange} placeholder={t('modelManager.selectModelType')}>
                <option value="main">{t('modelManager.main')}</option>
                <option value="lora">{t('modelManager.lora')}</option>
                <option value="controlnet">{t('modelManager.controlnet')}</option>
                <option value="vae">{t('modelManager.vae')}</option>
                <option value="ip_adapter">{t('modelManager.ipAdapter')}</option>
                <option value="t2i_adapter">{t('modelManager.t2iAdapter')}</option>
                <option value="control_lora">{t('modelManager.controlLora')}</option>
                <option value="embedding">{t('modelManager.embedding')}</option>
                <option value="spandrel_image_to_image">{t('modelManager.spandrel')}</option>
              </Select>
            </FormControl>

            <FormControl flex={1}>
              <FormLabel>{t('modelManager.baseModel')}</FormLabel>
              <Select value={config.base || ''} onChange={onBaseChange} placeholder={t('modelManager.selectBaseModel')}>
                <option value="sd-1">{t('modelManager.sd1')}</option>
                <option value="sd-2">{t('modelManager.sd2')}</option>
                <option value="sdxl">{t('modelManager.sdxl')}</option>
                <option value="sdxl-refiner">{t('modelManager.sdxlRefiner')}</option>
                <option value="sd-3">{t('modelManager.sd3')}</option>
                <option value="flux">{t('modelManager.flux')}</option>
                <option value="cogview4">{t('modelManager.cogview4')}</option>
                <option value="any">{t('modelManager.any')}</option>
              </Select>
            </FormControl>
          </Flex>

          <Flex gap={4}>
            <FormControl flex={1}>
              <FormLabel>{t('modelManager.format')}</FormLabel>
              <Select
                value={config.format || ''}
                onChange={onFormatChange}
                placeholder={t('modelManager.selectFormat')}
              >
                <option value="diffusers">{t('modelManager.diffusers')}</option>
                <option value="checkpoint">{t('modelManager.checkpoint')}</option>
                <option value="lycoris">{t('modelManager.lycoris')}</option>
                <option value="invokeai">{t('modelManager.invokeai')}</option>
                <option value="embed_file">{t('modelManager.embedFile')}</option>
                <option value="embed_folder">{t('modelManager.embedFolder')}</option>
                <option value="onnx">{t('modelManager.onnx')}</option>
                <option value="bnb_quantized_nf4b">{t('modelManager.bnbQuantized')}</option>
              </Select>
            </FormControl>

            <FormControl flex={1}>
              <FormLabel>{t('modelManager.predictionType')}</FormLabel>
              <Select
                value={config.prediction_type || ''}
                onChange={onPredictionTypeChange}
                placeholder={t('modelManager.selectPredictionType')}
              >
                <option value="epsilon">{t('modelManager.epsilon')}</option>
                <option value="v_prediction">{t('modelManager.vPrediction')}</option>
                <option value="sample">{t('modelManager.sample')}</option>
              </Select>
              <FormHelperText>{t('modelManager.predictionTypeHelper')}</FormHelperText>
            </FormControl>
          </Flex>

          <FormControl>
            <FormLabel>{t('modelManager.variant')}</FormLabel>
            <Select
              value={config.variant || ''}
              onChange={onVariantChange}
              placeholder={t('modelManager.selectVariant')}
            >
              <option value="normal">{t('modelManager.normal')}</option>
              <option value="inpaint">{t('modelManager.inpaint')}</option>
              <option value="depth">{t('modelManager.depth')}</option>
            </Select>
            <FormHelperText>{t('modelManager.variantHelper')}</FormHelperText>
          </FormControl>
        </Flex>
      </Flex>
    </Box>
  );
});

ManualModelConfigPanel.displayName = 'ManualModelConfigPanel';
