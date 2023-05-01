import { createSelector } from '@reduxjs/toolkit';

import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import IAINumberInput from 'common/components/IAINumberInput';
import { useEffect, useState } from 'react';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';

import {
  Flex,
  FormControl,
  FormLabel,
  HStack,
  Text,
  VStack,
} from '@chakra-ui/react';

// import { addNewModel } from 'app/socketio/actions';
import { Field, Formik } from 'formik';
import { useTranslation } from 'react-i18next';

import type { InvokeModelConfigProps } from 'app/types/invokeai';
import type { RootState } from 'app/store/store';
import type { FieldInputProps, FormikProps } from 'formik';
import { isEqual, pickBy } from 'lodash-es';
import ModelConvert from './ModelConvert';
import IAIFormHelperText from 'common/components/IAIForms/IAIFormHelperText';
import IAIFormErrorMessage from 'common/components/IAIForms/IAIFormErrorMessage';
import IAIForm from 'common/components/IAIForm';

const selector = createSelector(
  [systemSelector],
  (system) => {
    const { openModel, model_list } = system;
    return {
      model_list,
      openModel,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

const MIN_MODEL_SIZE = 64;
const MAX_MODEL_SIZE = 2048;

export default function CheckpointModelEdit() {
  const { openModel, model_list } = useAppSelector(selector);
  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  const [editModelFormValues, setEditModelFormValues] =
    useState<InvokeModelConfigProps>({
      name: '',
      description: '',
      config: 'configs/stable-diffusion/v1-inference.yaml',
      weights: '',
      vae: '',
      width: 512,
      height: 512,
      default: false,
      format: 'ckpt',
    });

  useEffect(() => {
    if (openModel) {
      const retrievedModel = pickBy(model_list, (_val, key) => {
        return isEqual(key, openModel);
      });
      setEditModelFormValues({
        name: openModel,
        description: retrievedModel[openModel]?.description,
        config: retrievedModel[openModel]?.config,
        weights: retrievedModel[openModel]?.weights,
        vae: retrievedModel[openModel]?.vae,
        width: retrievedModel[openModel]?.width,
        height: retrievedModel[openModel]?.height,
        default: retrievedModel[openModel]?.default,
        format: 'ckpt',
      });
    }
  }, [model_list, openModel]);

  const editModelFormSubmitHandler = (values: InvokeModelConfigProps) => {
    dispatch(
      addNewModel({
        ...values,
        width: Number(values.width),
        height: Number(values.height),
      })
    );
  };

  return openModel ? (
    <Flex flexDirection="column" rowGap={4} width="100%">
      <Flex alignItems="center" gap={4} justifyContent="space-between">
        <Text fontSize="lg" fontWeight="bold">
          {openModel}
        </Text>
        <ModelConvert model={openModel} />
      </Flex>
      <Flex
        flexDirection="column"
        maxHeight={window.innerHeight - 270}
        overflowY="scroll"
        paddingInlineEnd={8}
      >
        <Formik
          enableReinitialize={true}
          initialValues={editModelFormValues}
          onSubmit={editModelFormSubmitHandler}
        >
          {({ handleSubmit, errors, touched }) => (
            <IAIForm onSubmit={handleSubmit}>
              <VStack rowGap={2} alignItems="start">
                {/* Description */}
                <FormControl
                  isInvalid={!!errors.description && touched.description}
                  isRequired
                >
                  <FormLabel htmlFor="description" fontSize="sm">
                    {t('modelManager.description')}
                  </FormLabel>
                  <VStack alignItems="start">
                    <Field
                      as={IAIInput}
                      id="description"
                      name="description"
                      type="text"
                      width="full"
                    />
                    {!!errors.description && touched.description ? (
                      <IAIFormErrorMessage>
                        {errors.description}
                      </IAIFormErrorMessage>
                    ) : (
                      <IAIFormHelperText>
                        {t('modelManager.descriptionValidationMsg')}
                      </IAIFormHelperText>
                    )}
                  </VStack>
                </FormControl>

                {/* Config */}
                <FormControl
                  isInvalid={!!errors.config && touched.config}
                  isRequired
                >
                  <FormLabel htmlFor="config" fontSize="sm">
                    {t('modelManager.config')}
                  </FormLabel>
                  <VStack alignItems="start">
                    <Field
                      as={IAIInput}
                      id="config"
                      name="config"
                      type="text"
                      width="full"
                    />
                    {!!errors.config && touched.config ? (
                      <IAIFormErrorMessage>{errors.config}</IAIFormErrorMessage>
                    ) : (
                      <IAIFormHelperText>
                        {t('modelManager.configValidationMsg')}
                      </IAIFormHelperText>
                    )}
                  </VStack>
                </FormControl>

                {/* Weights */}
                <FormControl
                  isInvalid={!!errors.weights && touched.weights}
                  isRequired
                >
                  <FormLabel htmlFor="config" fontSize="sm">
                    {t('modelManager.modelLocation')}
                  </FormLabel>
                  <VStack alignItems="start">
                    <Field
                      as={IAIInput}
                      id="weights"
                      name="weights"
                      type="text"
                      width="full"
                    />
                    {!!errors.weights && touched.weights ? (
                      <IAIFormErrorMessage>
                        {errors.weights}
                      </IAIFormErrorMessage>
                    ) : (
                      <IAIFormHelperText>
                        {t('modelManager.modelLocationValidationMsg')}
                      </IAIFormHelperText>
                    )}
                  </VStack>
                </FormControl>

                {/* VAE */}
                <FormControl isInvalid={!!errors.vae && touched.vae}>
                  <FormLabel htmlFor="vae" fontSize="sm">
                    {t('modelManager.vaeLocation')}
                  </FormLabel>
                  <VStack alignItems="start">
                    <Field
                      as={IAIInput}
                      id="vae"
                      name="vae"
                      type="text"
                      width="full"
                    />
                    {!!errors.vae && touched.vae ? (
                      <IAIFormErrorMessage>{errors.vae}</IAIFormErrorMessage>
                    ) : (
                      <IAIFormHelperText>
                        {t('modelManager.vaeLocationValidationMsg')}
                      </IAIFormHelperText>
                    )}
                  </VStack>
                </FormControl>

                <HStack width="100%">
                  {/* Width */}
                  <FormControl isInvalid={!!errors.width && touched.width}>
                    <FormLabel htmlFor="width" fontSize="sm">
                      {t('modelManager.width')}
                    </FormLabel>
                    <VStack alignItems="start">
                      <Field id="width" name="width">
                        {({
                          field,
                          form,
                        }: {
                          field: FieldInputProps<number>;
                          form: FormikProps<InvokeModelConfigProps>;
                        }) => (
                          <IAINumberInput
                            id="width"
                            name="width"
                            min={MIN_MODEL_SIZE}
                            max={MAX_MODEL_SIZE}
                            step={64}
                            value={form.values.width}
                            onChange={(value) =>
                              form.setFieldValue(field.name, Number(value))
                            }
                          />
                        )}
                      </Field>

                      {!!errors.width && touched.width ? (
                        <IAIFormErrorMessage>
                          {errors.width}
                        </IAIFormErrorMessage>
                      ) : (
                        <IAIFormHelperText>
                          {t('modelManager.widthValidationMsg')}
                        </IAIFormHelperText>
                      )}
                    </VStack>
                  </FormControl>

                  {/* Height */}
                  <FormControl isInvalid={!!errors.height && touched.height}>
                    <FormLabel htmlFor="height" fontSize="sm">
                      {t('modelManager.height')}
                    </FormLabel>
                    <VStack alignItems="start">
                      <Field id="height" name="height">
                        {({
                          field,
                          form,
                        }: {
                          field: FieldInputProps<number>;
                          form: FormikProps<InvokeModelConfigProps>;
                        }) => (
                          <IAINumberInput
                            id="height"
                            name="height"
                            min={MIN_MODEL_SIZE}
                            max={MAX_MODEL_SIZE}
                            step={64}
                            value={form.values.height}
                            onChange={(value) =>
                              form.setFieldValue(field.name, Number(value))
                            }
                          />
                        )}
                      </Field>

                      {!!errors.height && touched.height ? (
                        <IAIFormErrorMessage>
                          {errors.height}
                        </IAIFormErrorMessage>
                      ) : (
                        <IAIFormHelperText>
                          {t('modelManager.heightValidationMsg')}
                        </IAIFormHelperText>
                      )}
                    </VStack>
                  </FormControl>
                </HStack>

                <IAIButton
                  type="submit"
                  className="modal-close-btn"
                  isLoading={isProcessing}
                >
                  {t('modelManager.updateModel')}
                </IAIButton>
              </VStack>
            </IAIForm>
          )}
        </Formik>
      </Flex>
    </Flex>
  ) : (
    <Flex
      sx={{
        width: '100%',
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: 'base',
        bg: 'base.900',
      }}
    >
      <Text fontWeight={500}>Pick A Model To Edit</Text>
    </Flex>
  );
}
