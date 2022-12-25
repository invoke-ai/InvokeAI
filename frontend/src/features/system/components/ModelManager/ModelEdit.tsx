import { createSelector } from '@reduxjs/toolkit';
import { Field, FieldInputProps, Formik, FormikProps } from 'formik';
import { RootState, useAppDispatch, useAppSelector } from 'app/store';
import { addNewModel,  } from 'app/socketio/actions';
import { InvokeModelConfigProps } from 'app/invokeai';
import { SystemState } from 'features/system/store/systemSlice';
import {
  Button,
  Flex,
  FormControl,
  FormErrorMessage,
  FormHelperText,
  FormLabel,
  HStack,
  Input,
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  Text,
  VStack,
} from '@chakra-ui/react';




export default function ModelEdit() {

  const MIN_MODEL_SIZE = 64;
  const MAX_MODEL_SIZE = 2048;
  const modelListSelector = createSelector(
    (state: RootState) => state.system,
    (system: SystemState) => {
      const models = _.map(system.model_list, (model, key) => {
        return { name: key, ...model };
      });
  
      const selectedModel = models.find((model) => model.name === system.openModel);
     
      return {
        openedModel: {
          name:selectedModel?.name,
          description:selectedModel?.description,
          config:selectedModel?.config,
          weights:selectedModel?.weights,
          vae:selectedModel?.vae,
          width:selectedModel?.width,
          height:selectedModel?.height,
          default:selectedModel?.default,
        }
      };
    }
  );
  
 
  const { openedModel } = useAppSelector(modelListSelector);


  const viewModelFormValues: InvokeModelConfigProps = {
    name: openedModel.name || '',
    description: openedModel.description || '',
    config: openedModel.config || '',
    weights: openedModel.weights || '',
    vae: openedModel.vae || '',
    width: openedModel.width || 0,
    height: openedModel.height || 0,
    default: openedModel.default || false,
  };

  const dispatch = useAppDispatch();
  
  const addModelFormSubmitHandler = (values: InvokeModelConfigProps) => {
    dispatch(addNewModel(values));
    onClose();
  };

  function hasWhiteSpace(s: string) {
    return /\\s/g.test(s);
  }

  function baseValidation(value: string) {
    let error;
    if (hasWhiteSpace(value)) error = 'Cannot use spaces';
    return error;
  }

  if (!openedModel) {
    return ( 
      <Flex> 'No model selected' </Flex>
    
  )} else {
    
    return (
    <Formik
      initialValues={viewModelFormValues}
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
                  readOnly
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
              Save Model
            </Button>
          </VStack>
        </form>
      )}
    </Formik>
)}
}
