import {
  Flex,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
  Text,
} from '@chakra-ui/react';
import IAIMantineMultiSelect from 'common/components/IAIMantineMultiSelect';
import { forEach } from 'lodash-es';
import { PropsWithChildren, useCallback, useMemo, useRef } from 'react';
import { useGetTextualInversionModelsQuery } from 'services/api/endpoints/models';
import { PARAMETERS_PANEL_WIDTH } from 'theme/util/constants';
import { RootState } from '../../../app/store/store';
import { useAppSelector } from '../../../app/store/storeHooks';
import IAIMantineSelectItemWithTooltip from '../../../common/components/IAIMantineSelectItemWithTooltip';

type EmbeddingSelectItem = {
  label: string;
  value: string;
  description?: string;
};

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

    const data: EmbeddingSelectItem[] = [];

    forEach(embeddingQueryData.entities, (embedding, _) => {
      if (!embedding) return;

      data.push({
        value: embedding.name,
        label: embedding.name,
        description: embedding.description,
        ...(currentMainModel?.base_model !== embedding.base_model
          ? { disabled: true, tooltip: 'Incompatible base model' }
          : {}),
      });
    });

    return data;
  }, [embeddingQueryData, currentMainModel?.base_model]);

  const handleChange = useCallback(
    (v: string[]) => {
      if (v.length === 0) {
        return;
      }

      onSelect(v[0]);
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
            <Flex sx={{ justifyContent: 'center', p: 2 }}>
              <Text
                sx={{ fontSize: 'sm', color: 'base.500', _dark: 'base.700' }}
              >
                No Embeddings Loaded
              </Text>
            </Flex>
          ) : (
            <IAIMantineMultiSelect
              inputRef={inputRef}
              placeholder={'Add Embedding'}
              value={[]}
              data={data}
              maxDropdownHeight={400}
              nothingFound="No Matching Embeddings"
              itemComponent={IAIMantineSelectItemWithTooltip}
              disabled={data.length === 0}
              filter={(value, selected, item: EmbeddingSelectItem) =>
                item.label.toLowerCase().includes(value.toLowerCase().trim()) ||
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

export default ParamEmbeddingPopover;
