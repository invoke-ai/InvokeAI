import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { Box, Flex, Spacer, Text } from '@invoke-ai/ui-library';
import ModelBaseBadge from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelBaseBadge';
import ModelImage from 'features/modelManagerV2/subpanels/ModelManagerPanel/ModelImage';
import { filesize } from 'filesize';
import { memo, useCallback } from 'react';
import type { AnyModelConfig } from 'services/api/types';

const itemSx: SystemStyleObject = {
  display: 'flex',
  flexDir: 'column',
  p: 2,
  cursor: 'pointer',
  borderRadius: 'base',
  '&[data-selected="true"]': {
    bg: 'base.700',
  },
  '&[data-disabled="true"]': {
    cursor: 'not-allowed',
    opacity: 0.5,
  },
};

export const ModelComboboxItem = memo(
  (props: {
    model: AnyModelConfig;
    setActive: (key: string) => void;
    onSelect: (key: string) => void;
    isSelected: boolean;
    isDisabled: boolean;
  }) => {
    const { model, setActive, onSelect, isDisabled, isSelected } = props;
    const onPointerMove = useCallback(() => {
      setActive(model.key);
    }, [model.key, setActive]);
    const onClick = useCallback(() => {
      onSelect(model.key);
    }, [model.key, onSelect]);
    return (
      <Box
        role="option"
        sx={itemSx}
        id={model.key}
        aria-disabled={isDisabled}
        aria-selected={isSelected}
        data-disabled={isDisabled}
        data-selected={isSelected}
        onPointerMove={isDisabled ? undefined : onPointerMove}
        onClick={isDisabled ? undefined : onClick}
      >
        <ModelComboboxItemContent model={model} />
      </Box>
    );
  }
);
ModelComboboxItem.displayName = 'ModelComboboxItem';

const ModelComboboxItemContent = memo(({ model }: { model: AnyModelConfig }) => {
  return (
    <Flex tabIndex={-1} gap={2}>
      <ModelImage image_url={model.cover_image} />
      <Flex flexDir="column" gap={2} flex={1}>
        <Flex gap={2} alignItems="center">
          <Text fontSize="sm" fontWeight="semibold">
            {model.name}
          </Text>
          <Spacer />
          <Text variant="subtext" fontStyle="italic">
            {filesize(model.file_size)}
          </Text>
          <ModelBaseBadge base={model.base} />
        </Flex>
        {model.description && <Text color="base.200">{model.description}</Text>}
      </Flex>
    </Flex>
  );
});
ModelComboboxItemContent.displayName = 'ModelComboboxItemContent';
