import type { DynamicPromptsSeedBehaviour } from '@features/generation/core/dynamicPrompts';
import type { SeedInputPatch } from '@platform/ui/SeedInput';
import type { ReactNode } from 'react';

import { Stack } from '@chakra-ui/react';
import { planSeedSubmission, type SeedMode } from '@platform/core/seed';
import { Field } from '@platform/ui/Field';
import { SeedInput } from '@platform/ui/SeedInput';

export interface SeedFieldProps {
  label: string;
  /** Validation error for the entered seed; shown only while the seed is in use. */
  error?: string | null;
  seed: number;
  seedMode: SeedMode;
  /** Iterations the next submission runs, which sizes the stepping-mode preview. */
  batchCount: number;
  /** Concrete prompts the next submission carries; one unless dynamic prompts expand it. */
  promptCount?: number;
  seedBehaviour?: DynamicPromptsSeedBehaviour;
  onCommit: (patch: SeedInputPatch) => void;
  /** Rows under the input, such as recent seeds. */
  children?: ReactNode;
}

/** The labelled seed row of the Generate-shaped widgets: the shared control under a field label, with its sequence sized by the widget's own batch. */
export const SeedField = ({
  batchCount,
  children,
  error,
  label,
  onCommit,
  promptCount = 1,
  seed,
  seedBehaviour = 'per-iteration',
  seedMode,
}: SeedFieldProps) => {
  const isRandom = seedMode === 'random';
  const plan =
    seedMode === 'increment' || seedMode === 'decrement'
      ? planSeedSubmission({ batchCount, promptCount, seedBehaviour, seedMode, startSeed: seed })
      : null;

  return (
    <Field error={isRandom ? undefined : error} hint="seed" label={label}>
      <Stack gap="1" w="full">
        <SeedInput ariaLabel={label} plan={plan} seed={seed} seedMode={seedMode} onCommit={onCommit} />
        {children}
      </Stack>
    </Field>
  );
};
