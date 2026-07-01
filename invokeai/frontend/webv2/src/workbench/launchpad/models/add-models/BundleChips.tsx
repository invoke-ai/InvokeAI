/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
import type { StarterModelBundle } from '@workbench/models/types';
import type { ReactNode } from 'react';

import { Badge, Box, HStack, Icon } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { CheckIcon, FolderIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

/** Horizontal row of bundle filters. */
export const BundleChips = ({
  bundles,
  onSelect,
  selectedName,
  starterCount,
  trailing,
}: {
  bundles: StarterModelBundle[];
  onSelect: (name: string | null) => void;
  selectedName: string | null;
  starterCount: number;
  trailing?: ReactNode;
}) => {
  const { t } = useTranslation();

  if (bundles.length === 0 && !trailing) {
    return null;
  }

  return (
    <HStack align="center" gap="2" minW="0" px="3" pb="1">
      {bundles.length > 0 ? (
        <HStack
          flex="1"
          gap="1.5"
          minW="0"
          overflowX="auto"
          css={{ scrollbarWidth: 'thin', '&::-webkit-scrollbar': { height: '4px' } }}
        >
          <BundleChip
            isSelected={selectedName === null}
            label={t('common.all')}
            subLabel={`${starterCount}`}
            onSelect={() => onSelect(null)}
          />
          {bundles.map((bundle) => {
            const missingCount = bundle.models.filter((model) => !model.is_installed).length;

            return (
              <BundleChip
                key={bundle.name}
                isComplete={missingCount === 0}
                isSelected={selectedName === bundle.name}
                label={bundle.name}
                subLabel={`${bundle.models.length}`}
                onSelect={() => onSelect(bundle.name)}
              />
            );
          })}
        </HStack>
      ) : (
        <Box flex="1" minW="0" />
      )}
      {trailing}
    </HStack>
  );
};

const BundleChip = ({
  isComplete = false,
  isSelected,
  label,
  onSelect,
  subLabel,
}: {
  isComplete?: boolean;
  isSelected: boolean;
  label: string;
  onSelect: () => void;
  subLabel: string;
}) => (
  <Button flexShrink={0} size="xs" variant={isSelected ? 'solid' : 'subtle'} onClick={onSelect}>
    <Icon as={isComplete ? CheckIcon : FolderIcon} boxSize="3" />
    {label}
    <Badge variant="surface" size="xs" ms="1" me="-1">
      {subLabel}
    </Badge>
  </Button>
);
/* eslint-disable react-perf/jsx-no-jsx-as-prop, react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-new-object-as-prop */
