import {
  Button,
  Flex,
  FormControl,
  FormErrorMessage,
  FormHelperText,
  FormLabel,
  HStack,
  Input,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Text,
  useDisclosure,
  VStack,
} from '@chakra-ui/react';

import React from 'react';
import { FaPlus } from 'react-icons/fa';
import { Field, FieldInputProps, Formik, FormikProps } from 'formik';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import { addNewModel } from 'app/socketio/actions';
import { InvokeModelConfigProps } from 'app/invokeai';
import IAICheckbox from 'common/components/IAICheckbox';
import IAIButton from 'common/components/IAIButton';
import SearchModels from './SearchModels';

const MIN_MODEL_SIZE = 64;
const MAX_MODEL_SIZE = 2048;

export default function AddModel() {
  const { isOpen, onOpen, onClose } = useDisclosure();
  const dispatch = useAppDispatch();

  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  function hasWhiteSpace(s: string) {
    return /\\s/g.test(s);
  }

  function baseValidation(value: string) {
    let error;
    if (hasWhiteSpace(value)) error = 'Cannot use spaces';
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
        aria-label="Add New Model"
        tooltip="Add New Model"
        onClick={onOpen}
        className="modal-close-btn"
        size={'sm'}
      >
        <Flex columnGap={'0.5rem'} alignItems="center">
          <FaPlus />
          Add New
        </Flex>
      </IAIButton>

      <Modal
        isOpen={isOpen}
        onClose={addModelModalClose}
        size="xl"
        closeOnOverlayClick={false}
      >
        <ModalOverlay />
        <ModalContent className="modal add-model-modal">
          <ModalHeader>Add New Model</ModalHeader>
          <ModalCloseButton />
          <ModalBody className="add-model-modal-body">
            <SearchModels />
            <IAICheckbox
              label="Add Manually"
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
                        Manual
                      </Text>
                      {/* Name */}
                      <FormControl
                        isInvalid={!!errors.name && touched.name}
                        isRequired
                      >
                        <FormLabel htmlFor="name">Name</FormLabel>
                        <VStack alignItems={'start'}>
                          <Field
                            as={Input}
                            id="name"
                            name="name"
                            type="text"
                            validate={baseValidation}
                          />
                          {!!errors.name && touched.name ? (
                            <FormErrorMessage>{errors.name}</FormErrorMessage>
                          ) : (
                            <FormHelperText margin={0}>
                              Enter a name for your model
                            </FormHelperText>
                          )}
                        </VStack>
                      </FormControl>

                      {/* Description */}
                      <FormControl
                        isInvalid={!!errors.description && touched.description}
                        isRequired
                      >
                        <FormLabel htmlFor="description">Description</FormLabel>
                        <VStack alignItems={'start'}>
                          <Field
                            as={Input}
                            id="description"
                            name="description"
                            type="text"
                          />
                          {!!errors.description && touched.description ? (
                            <FormErrorMessage>
                              {errors.description}
                            </FormErrorMessage>
                          ) : (
                            <FormHelperText margin={0}>
                              Add a description for your model
                            </FormHelperText>
                          )}
                        </VStack>
                      </FormControl>

                      {/* Config */}
                      <FormControl
                        isInvalid={!!errors.config && touched.config}
                        isRequired
                      >
                        <FormLabel htmlFor="config">Config</FormLabel>
                        <VStack alignItems={'start'}>
                          <Field
                            as={Input}
                            id="config"
                            name="config"
                            type="text"
                          />
                          {!!errors.config && touched.config ? (
                            <FormErrorMessage>{errors.config}</FormErrorMessage>
                          ) : (
                            <FormHelperText margin={0}>
                              Path to the config file of your model.
                            </FormHelperText>
                          )}
                        </VStack>
                      </FormControl>

                      {/* Weights */}
                      <FormControl
                        isInvalid={!!errors.weights && touched.weights}
                        isRequired
                      >
                        <FormLabel htmlFor="config">Model Location</FormLabel>
                        <VStack alignItems={'start'}>
                          <Field
                            as={Input}
                            id="weights"
                            name="weights"
                            type="text"
                          />
                          {!!errors.weights && touched.weights ? (
                            <FormErrorMessage>
                              {errors.weights}
                            </FormErrorMessage>
                          ) : (
                            <FormHelperText margin={0}>
                              Path to where your model is located.
                            </FormHelperText>
                          )}
                        </VStack>
                      </FormControl>

                      {/* VAE */}
                      <FormControl isInvalid={!!errors.vae && touched.vae}>
                        <FormLabel htmlFor="vae">VAE Location</FormLabel>
                        <VStack alignItems={'start'}>
                          <Field as={Input} id="vae" name="vae" type="text" />
                          {!!errors.vae && touched.vae ? (
                            <FormErrorMessage>{errors.vae}</FormErrorMessage>
                          ) : (
                            <FormHelperText margin={0}>
                              Path to where your VAE is located.
                            </FormHelperText>
                          )}
                        </VStack>
                      </FormControl>

                      <HStack width={'100%'}>
                        {/* Width */}
                        <FormControl
                          isInvalid={!!errors.width && touched.width}
                        >
                          <FormLabel htmlFor="width">Width</FormLabel>
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
                                    form.setFieldValue(
                                      field.name,
                                      Number(value)
                                    )
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
                              <FormErrorMessage>
                                {errors.width}
                              </FormErrorMessage>
                            ) : (
                              <FormHelperText margin={0}>
                                Default width of your model.
                              </FormHelperText>
                            )}
                          </VStack>
                        </FormControl>

                        {/* Height */}
                        <FormControl
                          isInvalid={!!errors.height && touched.height}
                        >
                          <FormLabel htmlFor="height">Height</FormLabel>
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
                                    form.setFieldValue(
                                      field.name,
                                      Number(value)
                                    )
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
                              <FormErrorMessage>
                                {errors.height}
                              </FormErrorMessage>
                            ) : (
                              <FormHelperText margin={0}>
                                Default height of your model.
                              </FormHelperText>
                            )}
                          </VStack>
                        </FormControl>
                      </HStack>

                      <Button
                        type="submit"
                        className="modal-close-btn"
                        isLoading={isProcessing}
                      >
                        Add Model
                      </Button>
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
