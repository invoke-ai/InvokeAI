import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { chakra, Checkbox, Flex, Spacer, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import {
  selectModelManagerV2Slice,
  selectSelectedModelKeys,
  setSelectedModelKey,
  toggleModelSelection,
} from 'features/modelManagerV2/store/modelManagerV2Slice';
import ModelBaseBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import ModelFormatBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelFormatBadge';
import { ModelDeleteButton } from 'features/modelManagerV2/subpanels/ModelPanel/ModelDeleteButton';
import { filesize } from 'filesize';
import type { ChangeEvent, MouseEvent } from 'react';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { AnyModelConfig } from 'services/api/types';

import ModelImage from './ModelImage';

const StyledLabel = chakra('label');

type ModelListItemProps = {
  model: AnyModelConfig;
};

const sx: SystemStyleObject = {
  paddingInlineStart: 10,
  paddingInlineEnd: 3,
  paddingBlock: 2,
  zIndex: 1,
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
  const { t } = useTranslation();
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
  const selectedModelKeys = useAppSelector(selectSelectedModelKeys);
  const isChecked = selectedModelKeys.includes(model.key);

  const handleRowClick = useCallback(
    (e: MouseEvent<HTMLDivElement>) => {
      // Ctrl/Cmd+Click toggles multi-selection
      if (e.ctrlKey || e.metaKey) {
        dispatch(toggleModelSelection(model.key));
      } else {
        dispatch(setSelectedModelKey(model.key));
      }
    },
    [model.key, dispatch]
  );

  const handleCheckboxChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      e.stopPropagation();
      dispatch(toggleModelSelection(model.key));
    },
    [model.key, dispatch]
  );

  const stopPropagation = useCallback((e: MouseEvent) => {
    e.stopPropagation();
  }, []);

  return (
    <Flex position="relative">
      <StyledLabel
        display="flex"
        alignItems="start"
        h="full"
        position="absolute"
        zIndex={2}
        paddingInlineStart={3}
        paddingInlineEnd={1}
        paddingBlock={3}
        cursor="pointer"
        onClick={stopPropagation}
      >
        <Checkbox aria-label={t('modelManager.selectModel')} isChecked={isChecked} onChange={handleCheckboxChange} />
      </StyledLabel>
      <Flex
        sx={sx}
        role="row"
        aria-selected={isSelected}
        justifyContent="flex-start"
        w="full"
        alignItems="flex-start"
        gap={2}
        cursor="pointer"
        onClick={handleRowClick}
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
    </Flex>
  );
};

export default memo(ModelListItem);
