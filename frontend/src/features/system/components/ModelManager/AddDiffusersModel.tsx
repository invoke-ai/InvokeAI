import {
  Flex,
  FormControl,
  FormErrorMessage,
  FormHelperText,
  FormLabel,
  VStack,
} from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIIconButton from 'common/components/IAIIconButton';
import IAIInput from 'common/components/IAIInput';
import { setAddNewModelUIOption } from 'features/options/store/optionsSlice';
import { Field, Formik } from 'formik';
import React from 'react';
import { useTranslation } from 'react-i18next';
import { BiArrowBack } from 'react-icons/bi';

import type { RootState } from 'app/store';
import { InvokeDiffusersModelConfigProps } from 'app/invokeai';
import { addNewModel } from 'app/socketio/actions';

export default function AddDiffusersModel() {
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

  const addModelFormValues: InvokeDiffusersModelConfigProps = {
    name: '',
    description: '',
    repo_id: '',
    format: 'diffusers',
    default: false,
    vae: {
      repo_id: '',
    },
  };

  const addModelFormSubmitHandler = (
    values: InvokeDiffusersModelConfigProps
  ) => {
    let diffusersModelToAdd = values;

    if (values.vae.repo_id == '') {
      diffusersModelToAdd = {
        ...diffusersModelToAdd,
        vae: { repo_id: values.repo_id + '/vae' },
      };
    }

    dispatch(addNewModel(diffusersModelToAdd));
    dispatch(setAddNewModelUIOption(null));
  };

  return (
    <Flex>
      <IAIIconButton
        aria-label={t('common:back')}
        tooltip={t('common:back')}
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
            <VStack rowGap={'0.5rem'}>
              {/* Name */}
              <FormControl isInvalid={!!errors.name && touched.name} isRequired>
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
                    <FormErrorMessage>{errors.description}</FormErrorMessage>
                  ) : (
                    <FormHelperText margin={0}>
                      {t('modelmanager:descriptionValidationMsg')}
                    </FormHelperText>
                  )}
                </VStack>
              </FormControl>

              {/* Repo ID */}
              <FormControl
                isInvalid={!!errors.repo_id && touched.repo_id}
                isRequired
              >
                <FormLabel htmlFor="repo_id" fontSize="sm">
                  {t('modelmanager:modelLocation')} /{' '}
                  {t('modelmanager:repo_id')}
                </FormLabel>
                <VStack alignItems={'start'}>
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
                      {t('modelmanager:modelLocationValidationMsg')}
                    </FormHelperText>
                  )}
                </VStack>
              </FormControl>

              {/* VAE */}
              <FormControl
                isInvalid={!!errors.vae?.repo_id && touched.vae?.repo_id}
                isRequired
              >
                <FormLabel htmlFor="vae.repo_id" fontSize="sm">
                  {t('modelmanager:vaeLocation')}
                </FormLabel>
                <VStack alignItems={'start'}>
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
                      {t('modelmanager:vaeLocationValidationMsg')}
                    </FormHelperText>
                  )}
                </VStack>
              </FormControl>

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
    </Flex>
  );
}
