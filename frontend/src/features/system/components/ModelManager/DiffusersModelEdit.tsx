import { createSelector } from '@reduxjs/toolkit';

import React, { useEffect, useState } from 'react';
import IAIInput from 'common/components/IAIInput';
import IAIButton from 'common/components/IAIButton';

import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';

import {
  Flex,
  FormControl,
  FormErrorMessage,
  FormHelperText,
  FormLabel,
  Text,
  VStack,
} from '@chakra-ui/react';

import { Field, Formik } from 'formik';
import { useTranslation } from 'react-i18next';
import { addNewModel } from 'app/socketio/actions';

import _ from 'lodash';

import type { RootState } from 'app/store';
import type { InvokeDiffusersModelConfigProps } from 'app/invokeai';

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

export default function DiffusersModelEdit() {
  const { openModel, model_list } = useAppSelector(selector);
  const isProcessing = useAppSelector(
    (state: RootState) => state.system.isProcessing
  );

  const dispatch = useAppDispatch();

  const { t } = useTranslation();

  const [editModelFormValues, setEditModelFormValues] =
    useState<InvokeDiffusersModelConfigProps>({
      name: '',
      description: '',
      repo_id: '',
      vae: '',
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
        repo_id: retrievedModel[openModel]?.repo_id,
        vae: retrievedModel[openModel]?.vae
          ? retrievedModel[openModel]?.vae
          : '',
        default: retrievedModel[openModel]?.default,
      });
    }
  }, [model_list, openModel]);

  const editModelFormSubmitHandler = (
    values: InvokeDiffusersModelConfigProps
  ) => {
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
                  <FormLabel htmlFor="description" fontSize="sm">
                    {t('modelmanager:description')}
                  </FormLabel>
                  <VStack alignItems={'start'}>
                    <Field
                      as={IAIInput}
                      id="description"
                      name="description"
                      type="text"
                      width="lg"
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
                  <FormLabel htmlFor="config" fontSize="sm">
                    {t('modelmanager:modelLocation')}
                  </FormLabel>
                  <VStack alignItems={'start'}>
                    <Field
                      as={IAIInput}
                      id="repo_id"
                      name="repo_id"
                      type="text"
                      width="lg"
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
                      width="lg"
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

                <IAIButton
                  type="submit"
                  className="modal-close-btn"
                  isLoading={isProcessing}
                >
                  {t('modelmanager:updateModel')}
                </IAIButton>
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
