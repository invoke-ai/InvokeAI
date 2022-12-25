import { createSelector, Dictionary } from '@reduxjs/toolkit';
import type { RootState } from 'app/store';
import { useAppSelector } from 'app/storeHooks';
import { systemSelector } from 'features/system/store/systemSelectors';
import React, { useEffect, useState } from 'react';
import _ from 'lodash';
import { Model } from 'app/invokeai';
import { Flex, Spacer, Text } from '@chakra-ui/react';

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

export default function ModelEdit() {
  const { openModel, model_list } = useAppSelector(selector);
  const [openedModel, setOpenedModel] = useState<Model>();

  useEffect(() => {
    if (openModel) {
      const retrievedModel = _.pickBy(model_list, (val, key) => {
        return _.isEqual(key, openModel);
      });
      setOpenedModel(retrievedModel[openModel]);
    }
  }, [model_list, openModel]);

  console.log(openedModel);

  return (
    <Flex flexDirection="column" rowGap="1rem">
      <Flex columnGap="1rem" alignItems="center">
        <Text fontSize="lg" fontWeight="bold">
          {openModel}
        </Text>
        <Text>{openedModel?.status}</Text>
      </Flex>
      <Flex flexDirection="column">
        <Text>{openedModel?.config}</Text>
        <Text>{openedModel?.description}</Text>
        <Text>{openedModel?.height}</Text>
        <Text>{openedModel?.width}</Text>
        <Text>{openedModel?.default}</Text>
        <Text>{openedModel?.vae}</Text>
        <Text>{openedModel?.weights}</Text>
      </Flex>
    </Flex>
  );
}
