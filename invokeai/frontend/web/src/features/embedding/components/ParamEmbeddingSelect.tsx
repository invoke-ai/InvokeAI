import { Flex, Text } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton from 'common/components/IAIButton';
import IAIMantineMultiSelect from 'common/components/IAIMantineMultiSelect';
import {
  setNegativePrompt,
  setPositivePrompt,
} from 'features/parameters/store/generationSlice';
import { forEach, join, map } from 'lodash-es';
import { forwardRef, useMemo, useState } from 'react';
import { useGetTextualInversionModelsQuery } from 'services/api/endpoints/models';

type EmbeddingSelectItem = {
  label: string;
  value: string;
  description?: string;
};

export default function ParamEmbeddingSelect() {
  const { data: embeddingQueryData } = useGetTextualInversionModelsQuery();
  const [selectedEmbeddings, setSelectedEmbeddings] = useState<
    string[] | undefined
  >(undefined);

  const dispatch = useAppDispatch();

  const positivePrompt = useAppSelector(
    (state: RootState) => state.generation.positivePrompt
  );

  const negativePrompt = useAppSelector(
    (state: RootState) => state.generation.negativePrompt
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
      });
    });

    return data;
  }, [embeddingQueryData]);

  const handlePositiveAdd = () => {
    if (!selectedEmbeddings) return;
    const parsedEmbeddings = join(
      map(selectedEmbeddings, (embedding) => `<${embedding}>`),
      ' '
    );
    dispatch(setPositivePrompt(`${positivePrompt} ${parsedEmbeddings}`));
    setSelectedEmbeddings([]);
  };

  const handleNegativeAdd = () => {
    if (!selectedEmbeddings) return;
    const parsedEmbeddings = join(
      map(selectedEmbeddings, (embedding) => `<${embedding}>`),
      ' '
    );
    dispatch(setNegativePrompt(`${negativePrompt} ${parsedEmbeddings}`));
    setSelectedEmbeddings([]);
  };

  return (
    <Flex gap={2} flexDirection="column">
      <IAIMantineMultiSelect
        placeholder="Pick Embedding"
        value={selectedEmbeddings}
        onChange={(v) => setSelectedEmbeddings(v)}
        data={data}
        maxDropdownHeight={400}
        nothingFound="No matching Embeddings"
        itemComponent={SelectItem}
        disabled={data.length === 0}
        filter={(value, selected, item: EmbeddingSelectItem) =>
          item.label.toLowerCase().includes(value.toLowerCase().trim()) ||
          item.value.toLowerCase().includes(value.toLowerCase().trim())
        }
        clearable
      />
      <Flex gap={2}>
        <IAIButton size="sm" w="100%" onClick={handlePositiveAdd}>
          Add To Positive
        </IAIButton>
        <IAIButton size="sm" w="100%" onClick={handleNegativeAdd}>
          Add To Negative
        </IAIButton>
      </Flex>
    </Flex>
  );
}

interface ItemProps extends React.ComponentPropsWithoutRef<'div'> {
  value: string;
  label: string;
  description?: string;
}

const SelectItem = forwardRef<HTMLDivElement, ItemProps>(
  ({ label, description, ...others }: ItemProps, ref) => {
    return (
      <div ref={ref} {...others}>
        <div>
          <Text>{label}</Text>
          {description && (
            <Text size="xs" color="base.600">
              {description}
            </Text>
          )}
        </div>
      </div>
    );
  }
);

SelectItem.displayName = 'SelectItem';
