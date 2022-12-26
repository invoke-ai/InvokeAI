import {
  Flex,
  FormControl,
  FormErrorMessage,
  FormHelperText,
  FormLabel,
  HStack,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  Text,
  useDisclosure,
  VStack,
} from '@chakra-ui/react';

import React from 'react';
import IAIInput from 'common/components/IAIInput';
import IAINumberInput from 'common/components/IAINumberInput';
import IAICheckbox from 'common/components/IAICheckbox';
import IAIButton from 'common/components/IAIButton';

import SearchModels from './SearchModels';

import { addNewModel } from 'app/socketio/actions';

import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { FaPlus } from 'react-icons/fa';
import { Field, Formik } from 'formik';
import { useTranslation } from 'react-i18next';

import type { FieldInputProps, FormikProps } from 'formik';
import type { RootState } from 'app/store';
import type { InvokeModelConfigProps } from 'app/invokeai';

const MIN_MODEL_SIZE = 64;
const MAX_MODEL_SIZE = 2048;

export default function AddModel() {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  function hasWhiteSpace(s: string) {
    return /\\s/g.test(s);
  }

  function baseValidation(value: string) {
    let error;
    if (hasWhiteSpace(value)) error = t('modelmanager:cannotUseSpaces');
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
    default: false,
  };

  const addModelFormSubmitHandler = (values: InvokeModelConfigProps) => {
    dispatch(addNewModel(values));
    onClose();
  };

  const addModelModalClose = () => {
    onClose();
  };

  const [addManually, setAddmanually] = React.useState<boolean>(false);

  return (
    <>
      <IAIButton
        aria-label={t('modelmanager:addNewModel')}
        tooltip={t('modelmanager:addNewModel')}
        onClick={onOpen}
        className="modal-close-btn"
        size={'sm'}
      >
        <Flex columnGap={'0.5rem'} alignItems="center">
          <FaPlus />
          {t('modelmanager:addNew')}
        </Flex>
      </IAIButton>

      <Modal
        isOpen={isOpen}
        onClose={addModelModalClose}
        size="3xl"
        closeOnOverlayClick={false}
      >
        <ModalOverlay />
        <ModalContent className="modal add-model-modal">
          <ModalHeader>{t('modelmanager:addNewModel')}</ModalHeader>
          <ModalCloseButton />
          <ModalBody className="add-model-modal-body">
            <SearchModels />
            <IAICheckbox
              label={t('modelmanager:addManually')}
              isChecked={addManually}
              onChange={() => setAddmanually(!addManually)}
            />

            {addManually && (
              <Formik
                initialValues={addModelFormValues}
                onSubmit={addModelFormSubmitHandler}
              >
                {({ handleSubmit, errors, touched }) => (
                  <form onSubmit={handleSubmit}>
                    <VStack rowGap={'0.5rem'}>
                      <Text fontSize={20} fontWeight="bold" alignSelf={'start'}>
                        {t('modelmanager:manual')}
                      </Text>
                      {/* Name */}
                      <FormControl
                        isInvalid={!!errors.name && touched.name}
                        isRequired
                      >
                        <FormLabel htmlFor="name" fontSize="sm">
                          {t('modelmanager:name')}
                        </FormLabel>
                        <VStack alignItems={'start'}>
                          <Field
                            as={IAIInput}
                            id="name"
                            name="name"
                            type="text"
                            validate={baseValidation}
                            width="2xl"
                          />
                          {!!errors.name && touched.name ? (
                            <FormErrorMessage>{errors.name}</FormErrorMessage>
                          ) : (
                            <FormHelperText margin={0}>
                              {t('modelmanager:nameValidationMsg')}
                            </FormHelperText>
                          )}
                        </VStack>
                      </FormControl>

                      {/* Description */}
                      <FormControl
                        isInvalid={!!errors.description && touched.description}
                        isRequired
                      >
                        <FormLabel htmlFor="description" fontSize="sm">
                          {t('modelmanager:description')}
                        </FormLabel>
                        <VStack alignItems={'start'}>
                          <Field
                            as={IAIInput}
                            id="description"
                            name="description"
                            type="text"
                            width="2xl"
                          />
                          {!!errors.description && touched.description ? (
                            <FormErrorMessage>
                              {errors.description}
                            </FormErrorMessage>
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
                        <FormLabel htmlFor="config" fontSize="sm">
                          {t('modelmanager:config')}
                        </FormLabel>
                        <VStack alignItems={'start'}>
                          <Field
                            as={IAIInput}
                            id="config"
                            name="config"
                            type="text"
                            width="2xl"
                          />
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
                        <FormLabel htmlFor="config" fontSize="sm">
                          {t('modelmanager:modelLocation')}
                        </FormLabel>
                        <VStack alignItems={'start'}>
                          <Field
                            as={IAIInput}
                            id="weights"
                            name="weights"
                            type="text"
                            width="2xl"
                          />
                          {!!errors.weights && touched.weights ? (
                            <FormErrorMessage>
                              {errors.weights}
                            </FormErrorMessage>
                          ) : (
                            <FormHelperText margin={0}>
                              {t('modelmanager:modelLocationValidationMsg')}
                            </FormHelperText>
                          )}
                        </VStack>
                      </FormControl>

                      {/* VAE */}
                      <FormControl isInvalid={!!errors.vae && touched.vae}>
                        <FormLabel htmlFor="vae" fontSize="sm">
                          {t('modelmanager:vaeLocation')}
                        </FormLabel>
                        <VStack alignItems={'start'}>
                          <Field
                            as={IAIInput}
                            id="vae"
                            name="vae"
                            type="text"
                            width="2xl"
                          />
                          {!!errors.vae && touched.vae ? (
                            <FormErrorMessage>{errors.vae}</FormErrorMessage>
                          ) : (
                            <FormHelperText margin={0}>
                              {t('modelmanager:vaeLocationValidationMsg')}
                            </FormHelperText>
                          )}
                        </VStack>
                      </FormControl>

                      <HStack width={'100%'}>
                        {/* Width */}
                        <FormControl
                          isInvalid={!!errors.width && touched.width}
                        >
                          <FormLabel htmlFor="width" fontSize="sm">
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
                                <IAINumberInput
                                  id="width"
                                  name="width"
                                  min={MIN_MODEL_SIZE}
                                  max={MAX_MODEL_SIZE}
                                  step={64}
                                  width="90%"
                                  value={form.values.width}
                                  onChange={(value) =>
                                    form.setFieldValue(
                                      field.name,
                                      Number(value)
                                    )
                                  }
                                />
                              )}
                            </Field>

                            {!!errors.width && touched.width ? (
                              <FormErrorMessage>
                                {errors.width}
                              </FormErrorMessage>
                            ) : (
                              <FormHelperText margin={0}>
                                {t('modelmanager:widthValidationMsg')}
                              </FormHelperText>
                            )}
                          </VStack>
                        </FormControl>

                        {/* Height */}
                        <FormControl
                          isInvalid={!!errors.height && touched.height}
                        >
                          <FormLabel htmlFor="height" fontSize="sm">
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
                                <IAINumberInput
                                  id="height"
                                  name="height"
                                  min={MIN_MODEL_SIZE}
                                  max={MAX_MODEL_SIZE}
                                  width="90%"
                                  step={64}
                                  value={form.values.height}
                                  onChange={(value) =>
                                    form.setFieldValue(
                                      field.name,
                                      Number(value)
                                    )
                                  }
                                />
                              )}
                            </Field>

                            {!!errors.height && touched.height ? (
                              <FormErrorMessage>
                                {errors.height}
                              </FormErrorMessage>
                            ) : (
                              <FormHelperText margin={0}>
                                {t('modelmanager:heightValidationMsg')}
                              </FormHelperText>
                            )}
                          </VStack>
                        </FormControl>
                      </HStack>

                      <IAIButton
                        type="submit"
                        className="modal-close-btn"
                        isLoading={isProcessing}
                      >
                        {t('modelmanager:addModel')}
                      </IAIButton>
                    </VStack>
                  </form>
                )}
              </Formik>
            )}
          </ModalBody>
        </ModalContent>
      </Modal>
    </>
  );
}
