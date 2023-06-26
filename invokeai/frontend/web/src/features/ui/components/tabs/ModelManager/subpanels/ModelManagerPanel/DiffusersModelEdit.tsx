import IAIButton from 'common/components/IAIButton';
import IAIInput from 'common/components/IAIInput';
import { useEffect, useState } from 'react';

import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';

import { Flex, FormControl, FormLabel, Text, VStack } from '@chakra-ui/react';

// import { addNewModel } from 'app/socketio/actions';
import { Field, Formik } from 'formik';
import { useTranslation } from 'react-i18next';

import type { RootState } from 'app/store/store';
import type { InvokeDiffusersModelConfigProps } from 'app/types/invokeai';
import IAIForm from 'common/components/IAIForm';
import IAIFormErrorMessage from 'common/components/IAIForms/IAIFormErrorMessage';
import IAIFormHelperText from 'common/components/IAIForms/IAIFormHelperText';

type DiffusersModelEditProps = {
  modelToEdit: string;
  retrievedModel: any;
};

export default function DiffusersModelEdit(props: DiffusersModelEditProps) {
  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const { retrievedModel, modelToEdit } = props;

  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  const [editModelFormValues, setEditModelFormValues] =
    useState<InvokeDiffusersModelConfigProps>({
      name: '',
      description: '',
      repo_id: '',
      path: '',
      vae: { repo_id: '', path: '' },
      default: false,
      model_format: 'diffusers',
    });

  useEffect(() => {
    setEditModelFormValues({
      name: modelToEdit,
      description: retrievedModel?.description,
      path:
        retrievedModel?.path && retrievedModel?.path !== 'None'
          ? retrievedModel?.path
          : '',
      repo_id:
        retrievedModel?.repo_id && retrievedModel?.repo_id !== 'None'
          ? retrievedModel?.repo_id
          : '',
      vae: {
        repo_id: retrievedModel?.vae?.repo_id
          ? retrievedModel?.vae?.repo_id
          : '',
        path: retrievedModel?.vae?.path ? retrievedModel?.vae?.path : '',
      },
      default: retrievedModel?.default,
      model_format: 'diffusers',
    });
  }, [retrievedModel, modelToEdit]);

  const editModelFormSubmitHandler = (
    values: InvokeDiffusersModelConfigProps
  ) => {
    const diffusersModelToEdit = values;

    if (values.path === '') delete diffusersModelToEdit.path;
    if (values.repo_id === '') delete diffusersModelToEdit.repo_id;
    if (values.vae.path === '') delete diffusersModelToEdit.vae.path;
    if (values.vae.repo_id === '') delete diffusersModelToEdit.vae.repo_id;

    dispatch(addNewModel(values));
  };

  return modelToEdit ? (
    <Flex flexDirection="column" rowGap={4} width="100%">
      <Flex alignItems="center">
        <Text fontSize="lg" fontWeight="bold">
          {retrievedModel.name}
        </Text>
      </Flex>
      <Flex flexDirection="column" overflowY="scroll" paddingInlineEnd={8}>
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

                {/* Path */}
                <FormControl
                  isInvalid={!!errors.path && touched.path}
                  isRequired
                >
                  <FormLabel htmlFor="path" fontSize="sm">
                    {t('modelManager.modelLocation')}
                  </FormLabel>
                  <VStack alignItems="start">
                    <Field
                      as={IAIInput}
                      id="path"
                      name="path"
                      type="text"
                      width="full"
                    />
                    {!!errors.path && touched.path ? (
                      <IAIFormErrorMessage>{errors.path}</IAIFormErrorMessage>
                    ) : (
                      <IAIFormHelperText>
                        {t('modelManager.modelLocationValidationMsg')}
                      </IAIFormHelperText>
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
                      width="full"
                    />
                    {!!errors.repo_id && touched.repo_id ? (
                      <IAIFormErrorMessage>
                        {errors.repo_id}
                      </IAIFormErrorMessage>
                    ) : (
                      <IAIFormHelperText>
                        {t('modelManager.repoIDValidationMsg')}
                      </IAIFormHelperText>
                    )}
                  </VStack>
                </FormControl>

                {/* VAE Path */}
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
                      width="full"
                    />
                    {!!errors.vae?.path && touched.vae?.path ? (
                      <IAIFormErrorMessage>
                        {errors.vae?.path}
                      </IAIFormErrorMessage>
                    ) : (
                      <IAIFormHelperText>
                        {t('modelManager.vaeLocationValidationMsg')}
                      </IAIFormHelperText>
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
                      width="full"
                    />
                    {!!errors.vae?.repo_id && touched.vae?.repo_id ? (
                      <IAIFormErrorMessage>
                        {errors.vae?.repo_id}
                      </IAIFormErrorMessage>
                    ) : (
                      <IAIFormHelperText>
                        {t('modelManager.vaeRepoIDValidationMsg')}
                      </IAIFormHelperText>
                    )}
                  </VStack>
                </FormControl>

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
      <Text fontWeight={'500'}>Pick A Model To Edit</Text>
    </Flex>
  );
}
