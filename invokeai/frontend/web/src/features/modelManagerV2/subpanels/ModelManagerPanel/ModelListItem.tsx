import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Flex, Spacer, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { selectModelManagerV2Slice, setSelectedModelKey } from 'features/modelManagerV2/store/modelManagerV2Slice';
import ModelBaseBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import ModelFormatBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelFormatBadge';
import { ModelDeleteButton } from 'features/modelManagerV2/subpanels/ModelPanel/ModelDeleteButton';
import { filesize } from 'filesize';
import { memo, useCallback, useMemo } from 'react';
import type { AnyModelConfig } from 'services/api/types';

import ModelImage from './ModelImage';

type ModelListItemProps = {
  model: AnyModelConfig;
};

const sx: SystemStyleObject = {
  paddingInline: 3,
  paddingBlock: 2,
  position: 'relative',
  rounded: 'base',
  '&:after,&:before': {
    content: `''`,
    position: 'absolute',
    pointerEvents: 'none',
  },
  '&:after': {
    h: '1px',
    bottom: '-0.5px',
    insetInline: 3,
    bg: 'base.850',
  },
  '&:before': {
    left: 1,
    w: 1,
    insetBlock: 2,
    rounded: 'base',
  },
  _hover: {
    bg: 'base.850',
    '& .delete-button': { opacity: 1 },
  },
  '& .delete-button': { opacity: 0 },
  "&[aria-selected='false']:hover:before": { bg: 'base.750' },
  "&[aria-selected='true']": {
    bg: 'base.800',
    '& .delete-button': { opacity: 1 },
  },
  "&[aria-selected='true']:before": { bg: 'invokeBlue.300' },
};

const ModelListItem = ({ model }: ModelListItemProps) => {
  const dispatch = useAppDispatch();
  const selectIsSelected = useMemo(
    () =>
      createSelector(
        selectModelManagerV2Slice,
        (modelManagerV2Slice) => modelManagerV2Slice.selectedModelKey === model.key
      ),
    [model.key]
  );
  const isSelected = useAppSelector(selectIsSelected);

  const handleSelectModel = useCallback(() => {
    dispatch(setSelectedModelKey(model.key));
  }, [model.key, dispatch]);

  return (
    <Flex
      sx={sx}
      aria-selected={isSelected}
      justifyContent="flex-start"
      w="full"
      alignItems="flex-start"
      gap={2}
      cursor="pointer"
      onClick={handleSelectModel}
    >
      <Flex gap={2} w="full" h="full" minW={0}>
        <ModelImage image_url={model.cover_image} />
        <Flex alignItems="flex-start" flexDir="column" w="full" minW={0}>
          <Flex gap={2} w="full" alignItems="flex-start">
            <Text fontWeight="semibold" noOfLines={1} wordBreak="break-all">
              {model.name}
            </Text>
            <Text variant="subtext" fontStyle="italic">
              {filesize(model.file_size)}
            </Text>
            <Spacer />
          </Flex>
          <Text variant="subtext" noOfLines={1}>
            {model.description || 'No Description'}
          </Text>
          <Flex gap={1} mt={1}>
            <ModelBaseBadge base={model.base} />
            <ModelFormatBadge format={model.format} />
          </Flex>
        </Flex>
      </Flex>
      <Flex mt={1}>
        <ModelDeleteButton modelConfig={model} showLabel={false} />
      </Flex>
    </Flex>
  );
};

export default memo(ModelListItem);
