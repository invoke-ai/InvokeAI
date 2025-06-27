import type { LoRA } from 'features/controlLayers/store/types';
import { MetadataItemView } from 'features/metadata/components/MetadataItemView';
import { RecallButton } from 'features/metadata/components/RecallButton';
import type { MetadataHandlers } from 'features/metadata/types';
import { handlers } from 'features/metadata/util/handlers';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { VStack, Text, Badge, HStack, IconButton, Tooltip } from '@invoke-ai/ui-library';
import { useClipboard } from 'common/hooks/useClipboard';
import { PiCopyBold } from 'react-icons/pi';

type Props = {
  metadata: unknown;
  /** Display mode for LoRA items */
  displayMode?: 'default' | 'badge';
  /** Whether to show copy functionality */
  showCopy?: boolean;
  /** Whether to show recall functionality */
  showRecall?: boolean;
};

export const MetadataLoRAs = ({ 
  metadata, 
  displayMode = 'default',
  showCopy = false,
  showRecall = true
}: Props) => {
  const [loras, setLoRAs] = useState<LoRA[]>([]);
  const clipboard = useClipboard();

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

  // Default display mode (original behavior)
  if (displayMode === 'default') {
    return (
      <>
        {loras.map((lora) => (
          <MetadataViewLoRA 
            key={lora.model.key} 
            label={label} 
            lora={lora} 
            handlers={handlers.loras}
            showRecall={showRecall}
          />
        ))}
      </>
    );
  }

  // Badge display mode (for staging area)
  if (displayMode === 'badge') {
    if (!loras || loras.length === 0) {
      return null;
    }

    return (
      <VStack align="start" spacing={3} w="full">
        <Text fontWeight="semibold" fontSize="md" color="base.100">LoRAs</Text>
        <VStack align="start" spacing={2} w="full">
          {loras.map((lora: LoRA, index: number) => (
            <BadgeLoRA 
              key={lora.id || index}
              lora={lora}
              index={index}
              handlers={handlers.loras}
              showCopy={showCopy}
              showRecall={showRecall}
            />
          ))}
        </VStack>
      </VStack>
    );
  }

  return null;
};

const MetadataViewLoRA = ({
  label,
  lora,
  handlers,
  showRecall = true,
}: {
  label: string;
  lora: LoRA;
  handlers: MetadataHandlers<LoRA[], LoRA>;
  showRecall?: boolean;
}) => {
  const onRecall = useCallback(() => {
    if (!handlers.recallItem || !showRecall) {
      return;
    }
    handlers.recallItem(lora, true).catch(() => {
      // no-op, the toast will show the error
    });
  }, [handlers, lora, showRecall]);

  const [renderedValue, setRenderedValue] = useState<React.ReactNode>(null);
  useEffect(() => {
    const _renderValue = async () => {
      if (!handlers.renderItemValue) {
        setRenderedValue(null);
        return;
      }
      const rendered = await handlers.renderItemValue(lora);
      setRenderedValue(rendered);
    };

    _renderValue();
  }, [handlers, lora]);

  return (
    <MetadataItemView 
      label={label} 
      isDisabled={false} 
      onRecall={showRecall ? onRecall : undefined} 
      renderedValue={renderedValue} 
    />
  );
};

const BadgeLoRA = ({
  lora,
  index,
  handlers,
  showCopy = false,
  showRecall = true,
}: {
  lora: LoRA;
  index: number;
  handlers: MetadataHandlers<LoRA[], LoRA>;
  showCopy?: boolean;
  showRecall?: boolean;
}) => {
  const [renderedValue, setRenderedValue] = useState<React.ReactNode>(null);
  const clipboard = useClipboard();

  useEffect(() => {
    const _renderValue = async () => {
      if (!handlers.renderItemValue) {
        setRenderedValue(`${lora.model.key} - ${lora.weight}`);
        return;
      }
      try {
        const rendered = await handlers.renderItemValue(lora);
        setRenderedValue(rendered);
      } catch {
        setRenderedValue(`${lora.model.key} - ${lora.weight}`);
      }
    };

    _renderValue();
  }, [handlers, lora]);

  const handleCopy = useCallback(() => {
    clipboard.writeText(`${lora.model.key} - ${lora.weight}`);
  }, [clipboard, lora]);

  const onRecall = useCallback(() => {
    if (!handlers.recallItem || !showRecall) {
      return;
    }
    handlers.recallItem(lora, true).catch(() => {
      // no-op, the toast will show the error
    });
  }, [handlers, lora, showRecall]);

  return (
    <VStack align="start" spacing={1} w="full">
      <Text fontSize="xs" fontWeight="medium" color="base.300" textTransform="uppercase" letterSpacing="wide">
        LoRA {index + 1}
      </Text>
      <VStack 
        position="relative" 
        w="full"
        _hover={{
          '& .hover-actions': {
            opacity: 1,
          }
        }}
      >
        <Badge 
          colorScheme="purple" 
          variant="subtle" 
          fontSize="sm" 
          px={3} 
          py={2}
          borderRadius="md"
          w="full"
          textAlign="center"
        >
          {renderedValue}
        </Badge>
        <HStack 
          className="hover-actions"
          position="absolute"
          top={-2}
          right={-2}
          opacity={0}
          transition="opacity 0.2s"
          spacing={1}
          bg="base.900"
          borderRadius="md"
          p={1}
          border="1px solid"
          borderColor="base.700"
          shadow="lg"
        >
          {showCopy && (
            <Tooltip label="Copy to clipboard">
              <IconButton
                size="xs"
                icon={<PiCopyBold />}
                onClick={handleCopy}
                colorScheme="base"
                variant="ghost"
                aria-label={`Copy LoRA ${index + 1} to clipboard`}
              />
            </Tooltip>
          )}
          {showRecall && handlers.recallItem && (
            <RecallButton
              label={handlers.getLabel()}
              onClick={onRecall}
              isDisabled={false}
            />
          )}
        </HStack>
      </VStack>
    </VStack>
  );
};
