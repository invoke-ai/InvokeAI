import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Alert, Flex, FormControlGroup, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectIsApiBaseModel } from 'features/controlLayers/store/paramsSlice';
import { memo } from 'react';

import { DimensionsAspectRatioSelect } from './DimensionsAspectRatioSelect';
import { DimensionsHeight } from './DimensionsHeight';
import { DimensionsLockAspectRatioButton } from './DimensionsLockAspectRatioButton';
import { DimensionsPreview } from './DimensionsPreview';
import { DimensionsSetOptimalSizeButton } from './DimensionsSetOptimalSizeButton';
import { DimensionsSwapButton } from './DimensionsSwapButton';
import { DimensionsWidth } from './DimensionsWidth';

export const Dimensions = memo(() => {
  const isApiModel = useAppSelector(selectIsApiBaseModel);

  return (
    <Flex gap={4} alignItems="center">
      <Flex gap={4} flexDirection="column" width="full">
        <FormControlGroup formLabelProps={formLabelProps}>
          <Flex gap={4}>
            <DimensionsAspectRatioSelect />
            <DimensionsSwapButton />
            {!isApiModel && (
              <>
                <DimensionsLockAspectRatioButton />
                <DimensionsSetOptimalSizeButton />
              </>
            )}
          </Flex>
          {!isApiModel && (
            <>
              <DimensionsWidth />
              <DimensionsHeight />
            </>
          )}
          {isApiModel && (
            <Alert status="info" borderRadius="base" flexDir="column" gap={2} overflow="unset">
              <Text fontSize="md" fontWeight="semibold">
                This model does not support pixel dimensions.
              </Text>
            </Alert>
          )}
        </FormControlGroup>
      </Flex>
      <Flex w="108px" h="108px" flexShrink={0} flexGrow={0} alignItems="center" justifyContent="center">
        <DimensionsPreview />
      </Flex>
    </Flex>
  );
});

Dimensions.displayName = 'Dimensions';

const formLabelProps: FormLabelProps = {
  minW: 10,
};
