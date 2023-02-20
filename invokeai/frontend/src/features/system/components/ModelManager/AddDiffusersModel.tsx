import {
  Flex,
  FormControl,
  FormErrorMessage,
  FormHelperText,
  FormLabel,
  Text,
  VStack,
} from '@chakra-ui/react';
import { InvokeDiffusersModelConfigProps } from 'app/invokeai';
import { addNewModel } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIInput from 'common/components/IAIInput';
import { setAddNewModelUIOption } from 'features/ui/store/uiSlice';
import { Field, Formik } from 'formik';
import { useTranslation } from 'react-i18next';
import { BiArrowBack } from 'react-icons/bi';

import type { RootState } from 'app/store';
import type { ReactElement } from 'react';

function FormItemWrapper({
  children,
}: {
  children: ReactElement | ReactElement[];
}) {
  return (
    <Flex
      flexDirection="column"
      backgroundColor="var(--background-color)"
      padding="1rem 1rem"
      borderRadius="0.5rem"
      rowGap="1rem"
      width="100%"
    >
      {children}
    </Flex>
  );
}

export default function AddDiffusersModel() {
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

  const addModelFormValues: InvokeDiffusersModelConfigProps = {
    name: '',
    description: '',
    repo_id: '',
    path: '',
    format: 'diffusers',
    default: false,
    vae: {
      repo_id: '',
      path: '',
    },
  };

  const addModelFormSubmitHandler = (
    values: InvokeDiffusersModelConfigProps
  ) => {
    const diffusersModelToAdd = values;

    if (values.path === '') delete diffusersModelToAdd.path;
    if (values.repo_id === '') delete diffusersModelToAdd.repo_id;
    if (values.vae.path === '') delete diffusersModelToAdd.vae.path;
    if (values.vae.repo_id === '') delete diffusersModelToAdd.vae.repo_id;

    dispatch(addNewModel(diffusersModelToAdd));
    dispatch(setAddNewModelUIOption(null));
  };

  return (
    <Flex>
      <IAIIconButton
        aria-label={t('common.back')}
        tooltip={t('common.back')}
        onClick={() => dispatch(setAddNewModelUIOption(null))}
        width="max-content"
        position="absolute"
        zIndex={1}
        size="sm"
        right={12}
        top={3}
        icon={<BiArrowBack />}
      />
      <Formik
        initialValues={addModelFormValues}
        onSubmit={addModelFormSubmitHandler}
      >
        {({ handleSubmit, errors, touched }) => (
          <form onSubmit={handleSubmit}>
            <VStack rowGap="0.5rem">
              <FormItemWrapper>
                {/* Name */}
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
                      width="2xl"
                      isRequired
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
              </FormItemWrapper>

              <FormItemWrapper>
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
                      width="2xl"
                      isRequired
                    />
                    {!!errors.description && touched.description ? (
                      <FormErrorMessage>{errors.description}</FormErrorMessage>
                    ) : (
                      <FormHelperText margin={0}>
                        {t('modelManager.descriptionValidationMsg')}
                      </FormHelperText>
                    )}
                  </VStack>
                </FormControl>
              </FormItemWrapper>

              <FormItemWrapper>
                <Text fontWeight="bold" fontSize="sm">
                  {t('modelManager.formMessageDiffusersModelLocation')}
                </Text>
                <Text
                  fontSize="sm"
                  fontStyle="italic"
                  color="var(--text-color-secondary)"
                >
                  {t('modelManager.formMessageDiffusersModelLocationDesc')}
                </Text>

                {/* Path */}
                <FormControl isInvalid={!!errors.path && touched.path}>
                  <FormLabel htmlFor="path" fontSize="sm">
                    {t('modelManager.modelLocation')}
                  </FormLabel>
                  <VStack alignItems="start">
                    <Field
                      as={IAIInput}
                      id="path"
                      name="path"
                      type="text"
                      width="2xl"
                    />
                    {!!errors.path && touched.path ? (
                      <FormErrorMessage>{errors.path}</FormErrorMessage>
                    ) : (
                      <FormHelperText margin={0}>
                        {t('modelManager.modelLocationValidationMsg')}
                      </FormHelperText>
                    )}
                  </VStack>
                </FormControl>

                {/* Repo ID */}
                <FormControl isInvalid={!!errors.repo_id && touched.repo_id}>
                  <FormLabel htmlFor="repo_id" fontSize="sm">
                    {t('modelManager.repo_id')}
                  </FormLabel>
                  <VStack alignItems="start">
                    <Field
                      as={IAIInput}
                      id="repo_id"
                      name="repo_id"
                      type="text"
                      width="2xl"
                    />
                    {!!errors.repo_id && touched.repo_id ? (
                      <FormErrorMessage>{errors.repo_id}</FormErrorMessage>
                    ) : (
                      <FormHelperText margin={0}>
                        {t('modelManager.repoIDValidationMsg')}
                      </FormHelperText>
                    )}
                  </VStack>
                </FormControl>
              </FormItemWrapper>

              <FormItemWrapper>
                {/* VAE Path */}
                <Text fontWeight="bold">
                  {t('modelManager.formMessageDiffusersVAELocation')}
                </Text>
                <Text
                  fontSize="sm"
                  fontStyle="italic"
                  color="var(--text-color-secondary)"
                >
                  {t('modelManager.formMessageDiffusersVAELocationDesc')}
                </Text>
                <FormControl
                  isInvalid={!!errors.vae?.path && touched.vae?.path}
                >
                  <FormLabel htmlFor="vae.path" fontSize="sm">
                    {t('modelManager.vaeLocation')}
                  </FormLabel>
                  <VStack alignItems="start">
                    <Field
                      as={IAIInput}
                      id="vae.path"
                      name="vae.path"
                      type="text"
                      width="2xl"
                    />
                    {!!errors.vae?.path && touched.vae?.path ? (
                      <FormErrorMessage>{errors.vae?.path}</FormErrorMessage>
                    ) : (
                      <FormHelperText margin={0}>
                        {t('modelManager.vaeLocationValidationMsg')}
                      </FormHelperText>
                    )}
                  </VStack>
                </FormControl>

                {/* VAE Repo ID */}
                <FormControl
                  isInvalid={!!errors.vae?.repo_id && touched.vae?.repo_id}
                >
                  <FormLabel htmlFor="vae.repo_id" fontSize="sm">
                    {t('modelManager.vaeRepoID')}
                  </FormLabel>
                  <VStack alignItems="start">
                    <Field
                      as={IAIInput}
                      id="vae.repo_id"
                      name="vae.repo_id"
                      type="text"
                      width="2xl"
                    />
                    {!!errors.vae?.repo_id && touched.vae?.repo_id ? (
                      <FormErrorMessage>{errors.vae?.repo_id}</FormErrorMessage>
                    ) : (
                      <FormHelperText margin={0}>
                        {t('modelManager.vaeRepoIDValidationMsg')}
                      </FormHelperText>
                    )}
                  </VStack>
                </FormControl>
              </FormItemWrapper>

              <IAIButton
                type="submit"
                className="modal-close-btn"
                isLoading={isProcessing}
              >
                {t('modelManager.addModel')}
              </IAIButton>
            </VStack>
          </form>
        )}
      </Formik>
    </Flex>
  );
}
