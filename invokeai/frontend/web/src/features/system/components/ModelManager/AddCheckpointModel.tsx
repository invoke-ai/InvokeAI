import {
  Flex,
  FormControl,
  FormErrorMessage,
  FormHelperText,
  FormLabel,
  HStack,
  Text,
  VStack,
} from '@chakra-ui/react';

import IAIButton from 'common/components/IAIButton';
import IAICheckbox from 'common/components/IAICheckbox';
import IAIInput from 'common/components/IAIInput';
import IAINumberInput from 'common/components/IAINumberInput';
import React from 'react';

import SearchModels from './SearchModels';

// import { addNewModel } from 'app/socketio/actions';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';

import { Field, Formik } from 'formik';
import { useTranslation } from 'react-i18next';

import type { InvokeModelConfigProps } from 'app/types/invokeai';
import type { RootState } from 'app/store/store';
import { setAddNewModelUIOption } from 'features/ui/store/uiSlice';
import type { FieldInputProps, FormikProps } from 'formik';
import IAIForm from 'common/components/IAIForm';
import { IAIFormItemWrapper } from 'common/components/IAIForms/IAIFormItemWrapper';

const MIN_MODEL_SIZE = 64;
const MAX_MODEL_SIZE = 2048;

export default function AddCheckpointModel() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  function hasWhiteSpace(s: string) {
    return /\s/.test(s);
  }

  function baseValidation(value: string) {
    let error;
    if (hasWhiteSpace(value)) error = t('modelManager.cannotUseSpaces');
    return error;
  }

  const addModelFormValues: InvokeModelConfigProps = {
    name: '',
    description: '',
    config: 'configs/stable-diffusion/v1-inference.yaml',
    weights: '',
    vae: '',
    width: 512,
    height: 512,
    format: 'ckpt',
    default: false,
  };

  const addModelFormSubmitHandler = (values: InvokeModelConfigProps) => {
    dispatch(addNewModel(values));
    dispatch(setAddNewModelUIOption(null));
  };

  const [addManually, setAddmanually] = React.useState<boolean>(false);

  return (
    <VStack gap={2} alignItems="flex-start">
      <Flex columnGap={4}>
        <IAICheckbox
          isChecked={!addManually}
          label={t('modelManager.scanForModels')}
          onChange={() => setAddmanually(!addManually)}
        />
        <IAICheckbox
          label={t('modelManager.addManually')}
          isChecked={addManually}
          onChange={() => setAddmanually(!addManually)}
        />
      </Flex>

      {addManually ? (
        <Formik
          initialValues={addModelFormValues}
          onSubmit={addModelFormSubmitHandler}
        >
          {({ handleSubmit, errors, touched }) => (
            <IAIForm onSubmit={handleSubmit} sx={{ w: 'full' }}>
              <VStack rowGap={2}>
                <Text fontSize={20} fontWeight="bold" alignSelf="start">
                  {t('modelManager.manual')}
                </Text>
                {/* Name */}
                <IAIFormItemWrapper>
                  <FormControl
                    isInvalid={!!errors.name && touched.name}
                    isRequired
                  >
                    <FormLabel htmlFor="name" fontSize="sm">
                      {t('modelManager.name')}
                    </FormLabel>
                    <VStack alignItems="start">
                      <Field
                        as={IAIInput}
                        id="name"
                        name="name"
                        type="text"
                        validate={baseValidation}
                        width="full"
                      />
                      {!!errors.name && touched.name ? (
                        <FormErrorMessage>{errors.name}</FormErrorMessage>
                      ) : (
                        <FormHelperText margin={0}>
                          {t('modelManager.nameValidationMsg')}
                        </FormHelperText>
                      )}
                    </VStack>
                  </FormControl>
                </IAIFormItemWrapper>

                {/* Description */}
                <IAIFormItemWrapper>
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
                        <FormErrorMessage>
                          {errors.description}
                        </FormErrorMessage>
                      ) : (
                        <FormHelperText margin={0}>
                          {t('modelManager.descriptionValidationMsg')}
                        </FormHelperText>
                      )}
                    </VStack>
                  </FormControl>
                </IAIFormItemWrapper>

                {/* Config */}
                <IAIFormItemWrapper>
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
                        <FormErrorMessage>{errors.config}</FormErrorMessage>
                      ) : (
                        <FormHelperText margin={0}>
                          {t('modelManager.configValidationMsg')}
                        </FormHelperText>
                      )}
                    </VStack>
                  </FormControl>
                </IAIFormItemWrapper>

                {/* Weights */}
                <IAIFormItemWrapper>
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
                        <FormErrorMessage>{errors.weights}</FormErrorMessage>
                      ) : (
                        <FormHelperText margin={0}>
                          {t('modelManager.modelLocationValidationMsg')}
                        </FormHelperText>
                      )}
                    </VStack>
                  </FormControl>
                </IAIFormItemWrapper>

                {/* VAE */}
                <IAIFormItemWrapper>
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
                        <FormErrorMessage>{errors.vae}</FormErrorMessage>
                      ) : (
                        <FormHelperText margin={0}>
                          {t('modelManager.vaeLocationValidationMsg')}
                        </FormHelperText>
                      )}
                    </VStack>
                  </FormControl>
                </IAIFormItemWrapper>

                <HStack width="100%">
                  {/* Width */}
                  <IAIFormItemWrapper>
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
                          <FormErrorMessage>{errors.width}</FormErrorMessage>
                        ) : (
                          <FormHelperText margin={0}>
                            {t('modelManager.widthValidationMsg')}
                          </FormHelperText>
                        )}
                      </VStack>
                    </FormControl>
                  </IAIFormItemWrapper>

                  {/* Height */}
                  <IAIFormItemWrapper>
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
                          <FormErrorMessage>{errors.height}</FormErrorMessage>
                        ) : (
                          <FormHelperText margin={0}>
                            {t('modelManager.heightValidationMsg')}
                          </FormHelperText>
                        )}
                      </VStack>
                    </FormControl>
                  </IAIFormItemWrapper>
                </HStack>

                <IAIButton
                  type="submit"
                  className="modal-close-btn"
                  isLoading={isProcessing}
                >
                  {t('modelManager.addModel')}
                </IAIButton>
              </VStack>
            </IAIForm>
          )}
        </Formik>
      ) : (
        <SearchModels />
      )}
    </VStack>
  );
}
