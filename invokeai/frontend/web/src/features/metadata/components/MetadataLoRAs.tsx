import { Text } from '@invoke-ai/ui-library';
import type { LoRA } from 'features/lora/store/loraSlice';
import { MetadataItemView } from 'features/metadata/components/MetadataItemView';
import type { MetadataHandlers } from 'features/metadata/types';
import { handlers } from 'features/metadata/util/handlers';
import { useCallback, useEffect, useMemo, useState } from 'react';

type Props = {
  metadata: unknown;
};

export const MetadataLoRAs = ({ metadata }: Props) => {
  const [loras, setLoRAs] = useState<LoRA[]>([]);

  useEffect(() => {
    const parse = async () => {
      try {
        const parsed = await handlers.loras.parse(metadata);
        setLoRAs(parsed);
      } catch (e) {
        setLoRAs([]);
      }
    };
    parse();
  }, [metadata]);

  const label = useMemo(() => handlers.loras.getLabel(), []);

  return (
    <>
      {loras.map((lora) => (
        <MetadataViewLoRA key={lora.model.key} label={label} lora={lora} handlers={handlers.loras} />
      ))}
    </>
  );
};

const MetadataViewLoRA = ({
  label,
  lora,
  handlers,
}: {
  label: string;
  lora: LoRA;
  handlers: MetadataHandlers<LoRA[], LoRA>;
}) => {
  const onRecall = useCallback(() => {
    if (!handlers.recallItem) {
      return;
    }
    handlers.recallItem(lora, true);
  }, [handlers, lora]);

  const renderedValue = useMemo(() => {
    if (!handlers.renderItemValue) {
      return null;
    }
    return <Text>{handlers.renderItemValue(lora)}</Text>;
  }, [handlers, lora]);

  return <MetadataItemView label={label} isDisabled={false} onRecall={onRecall} renderedValue={renderedValue} />;
};
