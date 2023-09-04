import {
  Flex,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Text,
} from '@chakra-ui/react';
import { SelectItem } from '@mantine/core';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import IAIMantineSearchableSelect from 'common/components/IAIMantineSearchableSelect';
import IAIMantineSelectItemWithTooltip from 'common/components/IAIMantineSelectItemWithTooltip';
import { MODEL_TYPE_MAP } from 'features/parameters/types/constants';
import { forEach } from 'lodash-es';
import { PropsWithChildren, memo, useCallback, useMemo, useRef } from 'react';
import { useGetTextualInversionModelsQuery } from 'services/api/endpoints/models';
import { PARAMETERS_PANEL_WIDTH } from 'theme/util/constants';

type Props = PropsWithChildren & {
  onSelect: (v: string) => void;
  isOpen: boolean;
  onClose: () => void;
};

const ParamEmbeddingPopover = (props: Props) => {
  const { onSelect, isOpen, onClose, children } = props;
  const { data: embeddingQueryData } = useGetTextualInversionModelsQuery();
  const inputRef = useRef<HTMLInputElement>(null);

  const currentMainModel = useAppSelector(
    (state: RootState) => state.generation.model
  );

  const data = useMemo(() => {
    if (!embeddingQueryData) {
      return [];
    }

    const data: SelectItem[] = [];

    forEach(embeddingQueryData.entities, (embedding, _) => {
      if (!embedding) {
        return;
      }

      const disabled = currentMainModel?.base_model !== embedding.base_model;

      data.push({
        value: embedding.model_name,
        label: embedding.model_name,
        group: MODEL_TYPE_MAP[embedding.base_model],
        disabled,
        tooltip: disabled
          ? `Incompatible base model: ${embedding.base_model}`
          : undefined,
      });
    });

    // Sort Alphabetically
    data.sort((a, b) =>
      a.label && b.label ? (a.label?.localeCompare(b.label) ? -1 : 1) : -1
    );

    return data.sort((a, b) => (a.disabled && !b.disabled ? 1 : -1));
  }, [embeddingQueryData, currentMainModel?.base_model]);

  const handleChange = useCallback(
    (v: string | null) => {
      if (!v) {
        return;
      }

      onSelect(v);
    },
    [onSelect]
  );

  return (
    <Popover
      initialFocusRef={inputRef}
      isOpen={isOpen}
      onClose={onClose}
      placement="bottom"
      openDelay={0}
      closeDelay={0}
      closeOnBlur={true}
      returnFocusOnClose={true}
    >
      <PopoverTrigger>{children}</PopoverTrigger>
      <PopoverContent
        sx={{
          p: 0,
          top: -1,
          shadow: 'dark-lg',
          borderColor: 'accent.300',
          borderWidth: '2px',
          borderStyle: 'solid',
          _dark: { borderColor: 'accent.400' },
        }}
      >
        <PopoverBody
          sx={{ p: 0, w: `calc(${PARAMETERS_PANEL_WIDTH} - 2rem )` }}
        >
          {data.length === 0 ? (
            <Flex
              sx={{
                justifyContent: 'center',
                p: 2,
                fontSize: 'sm',
                color: 'base.500',
                _dark: { color: 'base.700' },
              }}
            >
              <Text>No Embeddings Loaded</Text>
            </Flex>
          ) : (
            <IAIMantineSearchableSelect
              inputRef={inputRef}
              autoFocus
              placeholder="Add Embedding"
              value={null}
              data={data}
              nothingFound="No matching Embeddings"
              itemComponent={IAIMantineSelectItemWithTooltip}
              disabled={data.length === 0}
              onDropdownClose={onClose}
              filter={(value, item: SelectItem) =>
                item.label
                  ?.toLowerCase()
                  .includes(value.toLowerCase().trim()) ||
                item.value.toLowerCase().includes(value.toLowerCase().trim())
              }
              onChange={handleChange}
            />
          )}
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(ParamEmbeddingPopover);
