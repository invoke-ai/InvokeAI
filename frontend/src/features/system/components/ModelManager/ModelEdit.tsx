import { createSelector } from '@reduxjs/toolkit';
import type { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import React, { useEffect, useState } from 'react';
import _ from 'lodash';
import type { InvokeModelConfigProps } from 'app/invokeai';
import {
  Button,
  Flex,
  FormControl,
  FormErrorMessage,
  FormHelperText,
  FormLabel,
  Input,
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Text,
  VStack,
} from '@chakra-ui/react';
import { Field, Formik } from 'formik';
import type { FieldInputProps, FormikProps } from 'formik';
import { useTranslation } from 'react-i18next';
import { addNewModel } from 'app/socketio/actions';

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
      resultEqualityCheck: _.isEqual,
    },
  }
);

const MIN_MODEL_SIZE = 64;
const MAX_MODEL_SIZE = 2048;

export default function ModelEdit() {
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
    });

  useEffect(() => {
    if (openModel) {
      const retrievedModel = _.pickBy(model_list, (val, key) => {
        return _.isEqual(key, openModel);
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
      });
    }
  }, [model_list, openModel]);

  const editModelFormSubmitHandler = (values: InvokeModelConfigProps) => {
    dispatch(addNewModel(values));
  };

  return openModel ? (
    <Flex flexDirection="column" rowGap="1rem" width="100%">
      <Flex alignItems="center">
        <Text fontSize="lg" fontWeight="bold">
          {openModel}
        </Text>
      </Flex>
      <Flex
        flexDirection="column"
        maxHeight={window.innerHeight - 270}
        overflowY="scroll"
        paddingRight="2rem"
      >
        <Formik
          enableReinitialize={true}
          initialValues={editModelFormValues}
          onSubmit={editModelFormSubmitHandler}
        >
          {({ handleSubmit, errors, touched }) => (
            <form onSubmit={handleSubmit}>
              <VStack rowGap={'0.5rem'} alignItems="start">
                {/* Description */}
                <FormControl
                  isInvalid={!!errors.description && touched.description}
                  isRequired
                >
                  <FormLabel htmlFor="description">
                    {t('modelmanager:description')}
                  </FormLabel>
                  <VStack alignItems={'start'}>
                    <Field
                      as={Input}
                      id="description"
                      name="description"
                      type="text"
                    />
                    {!!errors.description && touched.description ? (
                      <FormErrorMessage>{errors.description}</FormErrorMessage>
                    ) : (
                      <FormHelperText margin={0}>
                        {t('modelmanager:descriptionValidationMsg')}
                      </FormHelperText>
                    )}
                  </VStack>
                </FormControl>

                {/* Config */}
                <FormControl
                  isInvalid={!!errors.config && touched.config}
                  isRequired
                >
                  <FormLabel htmlFor="config">
                    {t('modelmanager:config')}
                  </FormLabel>
                  <VStack alignItems={'start'}>
                    <Field as={Input} id="config" name="config" type="text" />
                    {!!errors.config && touched.config ? (
                      <FormErrorMessage>{errors.config}</FormErrorMessage>
                    ) : (
                      <FormHelperText margin={0}>
                        {t('modelmanager:configValidationMsg')}
                      </FormHelperText>
                    )}
                  </VStack>
                </FormControl>

                {/* Weights */}
                <FormControl
                  isInvalid={!!errors.weights && touched.weights}
                  isRequired
                >
                  <FormLabel htmlFor="config">
                    {t('modelmanager:modelLocation')}
                  </FormLabel>
                  <VStack alignItems={'start'}>
                    <Field as={Input} id="weights" name="weights" type="text" />
                    {!!errors.weights && touched.weights ? (
                      <FormErrorMessage>{errors.weights}</FormErrorMessage>
                    ) : (
                      <FormHelperText margin={0}>
                        {t('modelmanager:modelLocationValidationMsg')}
                      </FormHelperText>
                    )}
                  </VStack>
                </FormControl>

                {/* VAE */}
                <FormControl isInvalid={!!errors.vae && touched.vae}>
                  <FormLabel htmlFor="vae">
                    {t('modelmanager:vaeLocation')}
                  </FormLabel>
                  <VStack alignItems={'start'}>
                    <Field as={Input} id="vae" name="vae" type="text" />
                    {!!errors.vae && touched.vae ? (
                      <FormErrorMessage>{errors.vae}</FormErrorMessage>
                    ) : (
                      <FormHelperText margin={0}>
                        {t('modelmanager:vaeLocationValidationMsg')}
                      </FormHelperText>
                    )}
                  </VStack>
                </FormControl>

                <VStack width={'100%'}>
                  {/* Width */}
                  <FormControl isInvalid={!!errors.width && touched.width}>
                    <FormLabel htmlFor="width">
                      {t('modelmanager:width')}
                    </FormLabel>
                    <VStack alignItems={'start'}>
                      <Field id="width" name="width">
                        {({
                          field,
                          form,
                        }: {
                          field: FieldInputProps<number>;
                          form: FormikProps<InvokeModelConfigProps>;
                        }) => (
                          <NumberInput
                            {...field}
                            id="width"
                            name="width"
                            min={MIN_MODEL_SIZE}
                            max={MAX_MODEL_SIZE}
                            step={64}
                            onChange={(value) =>
                              form.setFieldValue(field.name, Number(value))
                            }
                          >
                            <NumberInputField />
                            <NumberInputStepper>
                              <NumberIncrementStepper />
                              <NumberDecrementStepper />
                            </NumberInputStepper>
                          </NumberInput>
                        )}
                      </Field>

                      {!!errors.width && touched.width ? (
                        <FormErrorMessage>{errors.width}</FormErrorMessage>
                      ) : (
                        <FormHelperText margin={0}>
                          {t('modelmanager:widthValidationMsg')}
                        </FormHelperText>
                      )}
                    </VStack>
                  </FormControl>

                  {/* Height */}
                  <FormControl isInvalid={!!errors.height && touched.height}>
                    <FormLabel htmlFor="height">
                      {t('modelmanager:height')}
                    </FormLabel>
                    <VStack alignItems={'start'}>
                      <Field id="height" name="height">
                        {({
                          field,
                          form,
                        }: {
                          field: FieldInputProps<number>;
                          form: FormikProps<InvokeModelConfigProps>;
                        }) => (
                          <NumberInput
                            {...field}
                            id="height"
                            name="height"
                            min={MIN_MODEL_SIZE}
                            max={MAX_MODEL_SIZE}
                            step={64}
                            onChange={(value) =>
                              form.setFieldValue(field.name, Number(value))
                            }
                          >
                            <NumberInputField />
                            <NumberInputStepper>
                              <NumberIncrementStepper />
                              <NumberDecrementStepper />
                            </NumberInputStepper>
                          </NumberInput>
                        )}
                      </Field>

                      {!!errors.height && touched.height ? (
                        <FormErrorMessage>{errors.height}</FormErrorMessage>
                      ) : (
                        <FormHelperText margin={0}>
                          {t('modelmanager:heightValidationMsg')}
                        </FormHelperText>
                      )}
                    </VStack>
                  </FormControl>
                </VStack>

                <Button
                  type="submit"
                  className="modal-close-btn"
                  isLoading={isProcessing}
                >
                  {t('modelmanager:updateModel')}
                </Button>
              </VStack>
            </form>
          )}
        </Formik>
      </Flex>
    </Flex>
  ) : (
    <Flex
      width="100%"
      height="250px"
      justifyContent="center"
      alignItems="center"
      backgroundColor="var(--background-color)"
      borderRadius="0.5rem"
    >
      <Text fontWeight="bold" color="var(--subtext-color-bright)">
        Pick A Model To Edit
      </Text>
    </Flex>
  );
}
